#!/usr/bin/env bash
# Upload already-packed pretrain/math-memory-full/v1 to edullm-landing (manifest last).
#SBATCH --job-name=p3math-mm-up
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=wheat-01
#SBATCH --output=logs/math_memory_full_up_%j.out
#SBATCH --error=logs/math_memory_full_up_%j.err

set -euo pipefail
PUB="${PUB:?}"
VERSION="${VERSION:-v1}"
DATASET_ID="pretrain/math-memory-full"
cd "$PUB"
mkdir -p logs

set -a
# shellcheck disable=SC1091
source "$PUB/aws-session.env"
set +a
unset AWS_PROFILE || true

VENV="${VENV:-$PUB/venv}"
PY="$VENV/bin/python"
DS_ROOT="$PUB/landing/$DATASET_ID/$VERSION"

"$PY" -c 'import boto3; print(boto3.client("sts", region_name="us-east-1").get_caller_identity()["Arn"])'
test -f "$DS_ROOT/dataset.json"
test -f "$DS_ROOT/tokens/manifest.json"
test -f "$DS_ROOT/build_meta.json"

export DS_ROOT VERSION DATASET_ID
"$PY" - <<'PY'
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
manifest = ds_root / "tokens" / "manifest.json"
print(f"payload_shards={len(payload)}", flush=True)
total = sum(p.stat().st_size for p in payload)
print(f"payload_bytes={total}", flush=True)

def upload(local: Path, key: str) -> None:
    t0 = time.time()
    size = local.stat().st_size
    print(f"[upload] start {key} bytes={size}", flush=True)
    s3.upload_file(str(local), bucket, key, Config=cfg)
    head = s3.head_object(Bucket=bucket, Key=key)
    assert int(head["ContentLength"]) == size
    dt = time.time() - t0
    mbps = (size / 1e6) / dt if dt > 0 else 0
    print(f"[upload] ok {key} seconds={dt:.1f} MBps={mbps:.1f}", flush=True)

t_all = time.time()
for path in payload:
    rel = path.relative_to(ds_root).as_posix()
    upload(path, f"{prefix}/{rel}")
upload(ds_root / "dataset.json", f"{prefix}/dataset.json")
upload(manifest, f"{prefix}/tokens/manifest.json")
print(f"[upload] sealed {prefix} total_seconds={time.time()-t_all:.1f}", flush=True)
PY
