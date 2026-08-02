#!/usr/bin/env bash
# Full 20B-token SmolLM2-135M fact-masked run. This is also the benchmark entrypoint.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/opt/edullm-venv}"
if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi
RUN_NAME="${RUN_NAME:-smollm2-135m-colmlm-$(date -u +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/workspace/${RUN_NAME}}"
AWS_ENV="${AWS_ENV:-/workspace/bootstrap/aws-session.env}"
WANDB_ENV="${WANDB_ENV:-/workspace/bootstrap/wandb-session.env}"
ANNOTATION_S3="${ANNOTATION_S3:-s3://edullm-checkpoints/runpod/colmlm-annotate/output}"
DATASET_META_S3="${DATASET_META_S3:-s3://edullm-data/pretrain/fineweb-edu-1b/v6}"
ANNOTATIONS_DIR="${RUN_DIR}/annotations"
DATASET_META_DIR="${RUN_DIR}/fineweb-edu-1b-v6-meta"
PACKED_DIR="${RUN_DIR}/packed"
OUTPUT_DIR="${RUN_DIR}/output"
NPROC="${NPROC:-4}"
# Keep tokens/step fixed across GPU-count migrations (4x40 == 8x20 == 160).
GLOBAL_BATCH_SAMPLES="${GLOBAL_BATCH_SAMPLES:-160}"
if [[ -z "${PER_DEVICE_BATCH_SIZE:-}" ]]; then
  if (( GLOBAL_BATCH_SAMPLES % NPROC != 0 )); then
    echo "GLOBAL_BATCH_SAMPLES=${GLOBAL_BATCH_SAMPLES} not divisible by NPROC=${NPROC}" >&2
    exit 2
  fi
  PER_DEVICE_BATCH_SIZE=$((GLOBAL_BATCH_SAMPLES / NPROC))
fi
CORPUS_MAX_TOKENS="${CORPUS_MAX_TOKENS:-750000000}"
NUM_EPOCHS="${NUM_EPOCHS:-27}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_EVERY="${LOG_EVERY:-20}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
COMPILE_MODE="${COMPILE_MODE:-max-autotune-no-cudagraphs}"

mkdir -p "${RUN_DIR}"/{logs,output,hf-cache} "${ANNOTATIONS_DIR}" "${DATASET_META_DIR}"
export HF_HOME="${RUN_DIR}/hf-cache"
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_MODULE_LOADING=LAZY
export TORCHINDUCTOR_CACHE_DIR="/opt/torchinductor-${RUN_NAME}"
export TRITON_CACHE_DIR="/opt/triton-${RUN_NAME}"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

cleanup_aws() {
  rm -f "${AWS_ENV}"
  unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
  unset AWS_PROFILE AWS_DEFAULT_PROFILE AWS_SESSION_EXPIRATION
}
trap cleanup_aws EXIT

python3 -c "import numpy, torch, transformers, wandb, zstandard"

if [[ "${RESUME_LOCAL:-0}" == "1" ]]; then
  echo "[stage] RESUME_LOCAL=1; skipping S3 download"
  if [[ -f "${PACKED_DIR}/_READY.json" ]]; then
    echo "[stage] packed corpus already present; annotation shards not required"
  else
    shards_found="$(find "${ANNOTATIONS_DIR}" -name '*.annotations.jsonl.zst' 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${shards_found}" != "19" ]]; then
      echo "RESUME_LOCAL without packed corpus requires 19 annotation shards under ${ANNOTATIONS_DIR}; found ${shards_found}" >&2
      exit 2
    fi
  fi
else
  if [[ ! -f "${AWS_ENV}" ]]; then
    echo "missing startup credential file ${AWS_ENV}" >&2
    exit 2
  fi
  command -v aws >/dev/null

  # The temporary AWS session is used only in this bounded block.
  # shellcheck disable=SC1090
  source "${AWS_ENV}"
  test -n "${AWS_ACCESS_KEY_ID:-}"
  test -n "${AWS_SECRET_ACCESS_KEY:-}"
  test -n "${AWS_SESSION_TOKEN:-}"
  echo "[stage] downloading annotation shards from ${ANNOTATION_S3}"
  aws s3 sync "${ANNOTATION_S3}/" "${ANNOTATIONS_DIR}/" \
    --exclude "*" --include "*.annotations.jsonl.zst" --include "*/_manifest.json" \
    --only-show-errors
  aws s3api list-objects-v2 \
    --bucket edullm-checkpoints \
    --prefix runpod/colmlm-annotate/output/ \
    --max-keys 100 > "${RUN_DIR}/annotation-s3-inventory.json"
  for name in dataset.json tokens/manifest.json vendor/manifest.json _VALIDATED.json; do
    mkdir -p "${DATASET_META_DIR}/$(dirname "${name}")"
    aws s3 cp "${DATASET_META_S3}/${name}" "${DATASET_META_DIR}/${name}" --only-show-errors
  done
fi

if [[ "${RESUME_LOCAL:-0}" != "1" ]]; then
python3 - \
  "${ANNOTATIONS_DIR}" \
  "${DATASET_META_DIR}" \
  "${SCRIPT_DIR}/annotation_inventory.json" \
  "${RUN_DIR}/annotation-s3-inventory.json" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
meta_root = Path(sys.argv[2])
expected_inventory = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
actual_inventory = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
actual_objects = {
    item["Key"]: (int(item["Size"]), item["ETag"].strip('"'))
    for item in actual_inventory.get("Contents", [])
}
shards = sorted(root.rglob("*.annotations.jsonl.zst"))
manifests = sorted(root.rglob("_manifest.json"))
if len(shards) != 19 or len(manifests) != 19:
    raise SystemExit(
        f"incomplete startup download: annotation_shards={len(shards)} manifests={len(manifests)}"
    )
if any(path.stat().st_size <= 0 for path in shards):
    raise SystemExit("one or more annotation shards are empty")
for item in expected_inventory["objects"]:
    actual = actual_objects.get(item["key"])
    expected = (int(item["bytes"]), item["etag"])
    if actual != expected:
        raise SystemExit(
            f"S3 identity mismatch for {item['key']}: expected={expected}, actual={actual}"
        )
    relative = item["key"].removeprefix(expected_inventory["prefix"])
    local = root / relative
    if local.stat().st_size != item["bytes"]:
        raise SystemExit(f"local size mismatch for {local}")
for index in range(19):
    worker = root / f"worker-{index}"
    shard = worker / f"train-{index:05d}.annotations.jsonl.zst"
    manifest_path = worker / "_manifest.json"
    expected_input = f"train-{index:05d}.jsonl.gz"
    if not shard.is_file() or not manifest_path.is_file():
        raise SystemExit(f"missing worker-{index} shard or manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("files") != [expected_input]:
        raise SystemExit(f"{manifest_path}: unexpected files {manifest.get('files')!r}")
    if manifest.get("workers") != {"w0": [expected_input]}:
        raise SystemExit(f"{manifest_path}: unexpected worker completion map")
dataset = json.loads((meta_root / "dataset.json").read_text(encoding="utf-8"))
if dataset.get("dataset_id") != "pretrain/fineweb-edu-1b":
    raise SystemExit("unexpected v6 dataset_id")
if dataset.get("version", {}).get("id") != "v6":
    raise SystemExit("unexpected FineWeb-Edu dataset version")
groups = {group["name"]: group for group in dataset.get("groups", [])}
if groups.get("tokens", {}).get("profile") != "pretrain-tokens/v1":
    raise SystemExit("v6 token group contract is missing")
if groups.get("vendor", {}).get("profile") != "vendored/v1":
    raise SystemExit("v6 raw vendor group contract is missing")
print(
    f"[stage] verified {len(shards)} annotation shards, "
    f"{sum(p.stat().st_size for p in shards):,} compressed bytes",
    flush=True,
)
PY

cleanup_aws
trap - EXIT
if env | awk -F= '/^AWS_(ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN|SECURITY_TOKEN)=/ {found=1} END {exit found ? 0 : 1}'; then
  echo "AWS credentials still present after startup staging" >&2
  exit 3
fi
echo "[stage] AWS credentials deleted; all remaining work is local scratch + W&B"
fi

if [[ -f "${WANDB_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${WANDB_ENV}"
fi
export WANDB_PROJECT="${WANDB_PROJECT:-edullm-smollm2-colmlm}"
export WANDB_GROUP="${WANDB_GROUP:-colmlm-fact-masked}"
export WANDB_MODE="${WANDB_MODE:-online}"
if [[ "${WANDB_MODE}" == "online" ]]; then
  test -n "${WANDB_API_KEY:-}" || {
    echo "WANDB_API_KEY missing from ${WANDB_ENV}" >&2
    exit 2
  }
fi

if [[ "${RESUME_LOCAL:-0}" == "1" && -f "${PACKED_DIR}/_READY.json" ]]; then
  echo "[prepare] packed corpus already ready; skipping"
else
echo "[prepare] tokenizing aligned annotation text and building masks (max ${CORPUS_MAX_TOKENS} tokens)"
python3 "${SCRIPT_DIR}/prepare_annotated_corpus.py" \
  --annotations-dir "${ANNOTATIONS_DIR}" \
  --output-dir "${PACKED_DIR}" \
  --seq-len 2048 \
  --expected-shards 19 \
  --max-tokens "${CORPUS_MAX_TOKENS}" \
  2>&1 | tee "${RUN_DIR}/logs/prepare.log"
fi

RESUME_ARGS=()
if [[ "${RESUME_FROM:-}" == "none" ]]; then
  echo "[train] fresh run; not resuming from checkpoints"
elif [[ -n "${RESUME_FROM:-}" ]]; then
  RESUME_ARGS+=(--resume-from "${RESUME_FROM}")
elif [[ -f "${OUTPUT_DIR}/latest_checkpoint.txt" ]]; then
  RESUME_FROM="$(tr -d '\r\n' < "${OUTPUT_DIR}/latest_checkpoint.txt")"
  if [[ -d "${RESUME_FROM}" ]]; then
    RESUME_ARGS+=(--resume-from "${RESUME_FROM}")
    echo "[train] resuming from ${RESUME_FROM}"
  fi
fi

nvidia-smi -L
echo "[train] full run ${RUN_NAME}; nproc=${NPROC} batch_per_device=${PER_DEVICE_BATCH_SIZE}"
python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
  "${SCRIPT_DIR}/train_smollm2_135m_colmlm_ddp.py" \
  --data-dir "${PACKED_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --seq-len 2048 \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --num-epochs "${NUM_EPOCHS}" \
  --max-train-tokens 20000000000 \
  --checkpoint-every-tokens 250000000 \
  --eval-interval-tokens 250000000 \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor 4 \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --compile \
  --compile-mode "${COMPILE_MODE}" \
  --no-gradient-checkpointing \
  --liger \
  --log-every "${LOG_EVERY}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-mode "${WANDB_MODE}" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "${RUN_DIR}/logs/train.log"
