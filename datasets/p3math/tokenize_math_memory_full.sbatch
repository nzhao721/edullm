#!/usr/bin/env bash
# Tokenize + pack only (no S3). Upload is a separate step after AWS session refresh.
#SBATCH --job-name=p3math-mm-tok
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=wheat-01
#SBATCH --output=logs/math_memory_full_tok_%j.out
#SBATCH --error=logs/math_memory_full_tok_%j.err

set -euo pipefail

P3="${P3:?}"
PUB="${PUB:?}"
STAGING="${STAGING:?}"
VERSION="${VERSION:-v1}"
DATASET_ID="pretrain/math-memory-full"
TOK_MANIFEST_SHA256="${TOK_MANIFEST_SHA256:-411c0206b78d13a6de912d12c5f0200bb6fa9bcba6f1424c21610fcd7f0af228}"

mkdir -p "$PUB/logs"
cd "$PUB"

VENV="${VENV:-$PUB/venv}"
# Prefer Lean publish venv (edullm_data + boto3); ensure zstandard present.
if ! "$VENV/bin/python" -c 'import zstandard' 2>/dev/null; then
  "$VENV/bin/pip" install -q zstandard
fi
export PYTHONPATH="$STAGING/datasets${PYTHONPATH:+:$PYTHONPATH}"
PY="$VENV/bin/python"

LANDING_ROOT="$PUB/landing"
DS_ROOT="$LANDING_ROOT/$DATASET_ID/$VERSION"
rm -rf "$DS_ROOT"
mkdir -p "$DS_ROOT"

echo "[1/2] tokenize → $DS_ROOT"
"$PY" "$STAGING/datasets/p3math/tokenize_p3math_bytes.py" \
  --scratch-root "$P3" \
  --out-root "$DS_ROOT" \
  --shard-tokens 500000000 \
  --val-fraction 0.0015

echo "[2/2] pack manifests (in-place)"
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

"$PY" - <<PY
import json
from pathlib import Path
meta = json.loads(Path("$DS_ROOT/build_meta.json").read_text())
print("BUILD_META_TRAIN", meta.get("train_tokens"), "VAL", meta.get("val_tokens"))
print("SHARDS", sorted(str(p) for p in Path("$DS_ROOT/tokens").rglob("*.u32le.bin"))[:20], "...")
PY

echo "[done] tokenize+pack complete; upload deferred until fresh aws-session.env"
