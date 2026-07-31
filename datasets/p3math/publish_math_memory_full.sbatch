#!/usr/bin/env bash
# Tokenize full P3Math mix → pack pretrain/math-memory-full/v1 → upload to edullm-landing.
#SBATCH --job-name=p3math-mm-full
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=wheat-01
#SBATCH --output=logs/math_memory_full_%j.out
#SBATCH --error=logs/math_memory_full_%j.err

set -euo pipefail

P3="${P3:?set P3 to p3math scratch root}"
PUB="${PUB:?set PUB to publish run dir}"
STAGING="${STAGING:?set STAGING to edullm farmshare staging}"
VERSION="${VERSION:-v1}"
DATASET_ID="pretrain/math-memory-full"
TOK_MANIFEST_SHA256="${TOK_MANIFEST_SHA256:-411c0206b78d13a6de912d12c5f0200bb6fa9bcba6f1424c21610fcd7f0af228}"

mkdir -p "$PUB/logs" "$P3/logs"
cd "$PUB"

set -a
# shellcheck disable=SC1091
source "$PUB/aws-session.env"
set +a
unset AWS_PROFILE || true

VENV="${VENV:-$PUB/venv}"
export PYTHONPATH="$STAGING/datasets${PYTHONPATH:+:$PYTHONPATH}"
PY="$VENV/bin/python"

"$PY" -c 'import boto3; print(boto3.client("sts", region_name="us-east-1").get_caller_identity())'

LANDING_ROOT="$PUB/landing"
DS_ROOT="$LANDING_ROOT/$DATASET_ID/$VERSION"
rm -rf "$DS_ROOT"
mkdir -p "$DS_ROOT"

echo "[1/3] tokenize → $DS_ROOT"
"$PY" "$STAGING/datasets/p3math/tokenize_p3math_bytes.py" \
  --scratch-root "$P3" \
  --out-root "$DS_ROOT" \
  --shard-tokens 500000000 \
  --val-fraction 0.0015

echo "[2/3] pack manifests (in-place)"
"$PY" "$STAGING/datasets/p3math/pack_edullm_landing.py" pretrain \
  --tokens-dir "$DS_ROOT/tokens" \
  --out-root "$LANDING_ROOT" \
  --in-place \
  --version "$VERSION" \
  --dataset-id "$DATASET_ID" \
  --purpose "Full math reasoning dataset for pretraining from scratch for the memory split P3 team" \
  --about "P3Math mix as raw UTF-8 byte tokens: filtered OpenWebMath, AlgebraicStack (formal+TeX), arXiv math-pure (cross-list purity + pure-math subcategory allowlist), and Lean4-Mathlib. Tokenizer tokenizer/bytes-utf8/v1. Held-out val is 0.15% of Lean bytes only; train holds the remaining Lean plus all other sources. No Gate A alphabet markers." \
  --notes "Val is Lean-only. Tokenization: each UTF-8 byte → uint32 id. Docs joined with blank lines." \
  --sources-json "$STAGING/datasets/p3math/math_memory_full_sources.json" \
  --tokenizer-id tokenizer/bytes-utf8 \
  --tokenizer-version v1 \
  --tokenizer-manifest-sha256 "$TOK_MANIFEST_SHA256" \
  --build-meta "$DS_ROOT/build_meta.json"

# Enrich sources[] token counts from build_meta into a note file (dataset.json already sealed).
export DS_ROOT
"$PY" - <<'PY'
import json
from pathlib import Path
import os
ds = Path(os.environ["DS_ROOT"])
meta = json.loads((ds / "build_meta.json").read_text())
print("BUILD_META_TRAIN", meta.get("train_tokens"), "VAL", meta.get("val_tokens"))
PY

echo "[3/3] upload to s3://edullm-landing (manifest last)"
export DS_ROOT VERSION DATASET_ID
DS_ROOT="$DS_ROOT" VERSION="$VERSION" DATASET_ID="$DATASET_ID" "$PY" - <<'PY'
import json, os, time
from pathlib import Path
import boto3
from boto3.s3.transfer import TransferConfig

ds_root = Path(os.environ["DS_ROOT"])
prefix = f"{os.environ['DATASET_ID']}/{os.environ['VERSION']}"
bucket = "edullm-landing"
s3 = boto3.client("s3", region_name="us-east-1")
cfg = TransferConfig(
    multipart_threshold=64 * 1024 * 1024,
    multipart_chunksize=64 * 1024 * 1024,
    max_concurrency=8,
    use_threads=True,
)

meta = json.loads((ds_root / "build_meta.json").read_text())
assert meta.get("alphabet_marker_tokens", 0) == 0, meta
assert meta.get("val_tokens", 0) > 0, meta
print("meta_ok", meta["train_tokens"], meta["val_tokens"], flush=True)

payload = sorted(ds_root.rglob("*.u32le.bin"))
meta_files = [ds_root / "dataset.json"]
manifest = ds_root / "tokens" / "manifest.json"
assert manifest.is_file(), manifest
assert (ds_root / "dataset.json").is_file()

def upload(local: Path, key: str) -> None:
    t0 = time.time()
    size = local.stat().st_size
    print(f"[upload] start {key} bytes={size}", flush=True)
    s3.upload_file(str(local), bucket, key, Config=cfg)
    head = s3.head_object(Bucket=bucket, Key=key)
    assert int(head["ContentLength"]) == size
    print(f"[upload] ok {key} seconds={time.time()-t0:.1f}", flush=True)

# payload shards first, then dataset.json, seal with tokens/manifest.json last
for path in payload:
    rel = path.relative_to(ds_root).as_posix()
    upload(path, f"{prefix}/{rel}")
upload(ds_root / "dataset.json", f"{prefix}/dataset.json")
upload(manifest, f"{prefix}/tokens/manifest.json")
print("[upload] sealed", prefix, flush=True)
PY

echo "[done] waiting for Gate A; check landing for _VALIDATED.json"
