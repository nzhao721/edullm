#!/usr/bin/env bash
# RunPod smoke: annotate a small FineWeb-Edu raw shard with ModernBERT tagger.
set -Eeuo pipefail

ROOT="${ROOT:-/workspace/colmlm_annotate}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/model/final}"
INPUT_DIR="${INPUT_DIR:-${ROOT}/input}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/output}"
BATCH="${BATCH:-32}"
MAX_FILES="${MAX_FILES:-1}"
MAX_DOCS_PER_FILE="${MAX_DOCS_PER_FILE:-200}"
WORKER_INDEX="${WORKER_INDEX:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"

export PYTHONUNBUFFERED=1

echo "[env] torch / cuda probe"
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("bf16", torch.cuda.is_bf16_supported())
PY

echo "[deps] transformers>=4.48 zstandard"
python3 -m pip install -q -U "transformers>=4.48" zstandard

mkdir -p "${OUTPUT_DIR}"
test -f "${MODEL_DIR}/config.json"
test -d "${INPUT_DIR}"

echo "[annotate] model=${MODEL_DIR} input=${INPUT_DIR} output=${OUTPUT_DIR}"
python3 "${ROOT}/annotate_modernbert.py" \
  --model-dir "${MODEL_DIR}" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --id-field doc_id \
  --batch "${BATCH}" \
  --max-files "${MAX_FILES}" \
  --max-docs-per-file "${MAX_DOCS_PER_FILE}" \
  --worker-index "${WORKER_INDEX}" \
  --num-workers "${NUM_WORKERS}" \
  --verify

echo "SMOKE_DONE output=${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
