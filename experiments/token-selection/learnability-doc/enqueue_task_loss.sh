#!/usr/bin/env bash
# Optional FarmShare helper: spawn task_loss_bpb for one checkpoint.
# Prefer the trainer's shared token_selection.olmo_ext.task_loss_hook (auto).
# Args: <step> <checkpoint_dir> <out_json> <run_id>
set -Eeuo pipefail

STEP="${1:?step}"
CKPT="${2:?checkpoint_dir}"
OUT="${3:?out_json}"
RUN_ID="${4:?run_id}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EVAL_PY="${TASK_LOSS_EVAL_SCRIPT:-${REPO_ROOT}/scripts/farmshare/task_loss/eval_task_loss_olmo_core.py}"
BASE_CONFIG="${LADDER_BASE_CONFIG:-${BASE_CONFIG:-}}"

mkdir -p "$(dirname "${OUT}")"
LOG_DIR="${TASK_LOSS_LOG_DIR:-$(dirname "${OUT}")/logs}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/step${STEP}_task_loss.log"

CMD=(python "${EVAL_PY}" --checkpoint "${CKPT}" --out "${OUT}" --run-name "${RUN_ID}" --format auto)
if [[ -n "${BASE_CONFIG}" ]]; then
  CMD+=(--base-config "${BASE_CONFIG}")
fi

# Prefer torchrun when TASK_LOSS_NPROC/NPROC > 1; else single process.
NPROC="${TASK_LOSS_NPROC:-1}"
if [[ "${NPROC}" -gt 1 ]]; then
  RUN=(torchrun --standalone --nproc_per_node="${NPROC}" "${CMD[@]}")
else
  RUN=("${CMD[@]}")
fi

nohup "${RUN[@]}" >"${LOG}" 2>&1 &
echo "enqueued step=${STEP} pid=$! log=${LOG} out=${OUT}"
