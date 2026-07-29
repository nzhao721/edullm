#!/usr/bin/env bash
# Launch Middle-PPL document CE (plain CE on filtered corpus) on 1..N GPUs.
#
# Required env / args (no host-specific defaults):
#   TRAIN_PATHS_FILE  — memmap path list from prepare_data.py
#   SAVE_FOLDER       — checkpoint root
#   PROGRESS_DIR      — metrics / run_meta
#
# Optional:
#   NPROC             — processes per node (default: 1, or WORLD_SIZE)
#   NAME              — run id (default: edullm-370M-middle-ppl-doc-ladder125-v1)
#   LENGTH_TOKENS     — default 10000000000 → 2384 steps
#   TASK_LOSS_EVAL    — 0 to disable post-save evals (default: enabled by trainer)
#   EXTRA_ARGS        — extra flags forwarded to the trainer
#
# Does not call AWS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/experiments/token-selection${PYTHONPATH:+:${PYTHONPATH}}"

: "${TRAIN_PATHS_FILE:?Set TRAIN_PATHS_FILE to the filtered paths list}"
: "${SAVE_FOLDER:?Set SAVE_FOLDER to the checkpoint directory}"
: "${PROGRESS_DIR:?Set PROGRESS_DIR to the progress/metrics directory}"

NAME="${NAME:-edullm-370M-middle-ppl-doc-ladder125-v1}"
LENGTH_TOKENS="${LENGTH_TOKENS:-10000000000}"

if [[ -n "${NPROC:-}" ]]; then
  NPROC_PER_NODE="${NPROC}"
elif [[ -n "${WORLD_SIZE:-}" ]]; then
  NPROC_PER_NODE="${WORLD_SIZE}"
else
  NPROC_PER_NODE=1
fi

if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC/WORLD_SIZE must be a positive integer, got: ${NPROC_PER_NODE}" >&2
  exit 1
fi

TRAINER="${SCRIPT_DIR}/train_ce_middle_ppl_doc.py"
COMMON_ARGS=(
  --name "${NAME}"
  --train-paths-file "${TRAIN_PATHS_FILE}"
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --length-tokens "${LENGTH_TOKENS}"
)

# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS:-} )

if [[ -n "${TASK_LOSS_OUT_DIR:-}" ]]; then
  EXTRA+=(--task-loss-results-dir "${TASK_LOSS_OUT_DIR}")
fi
if [[ -n "${TASK_LOSS_EVAL_SCRIPT:-}" ]]; then
  EXTRA+=(--task-loss-eval-script "${TASK_LOSS_EVAL_SCRIPT}")
fi
if [[ "${FRESH:-0}" == "1" ]]; then
  EXTRA+=(--fresh)
fi
if [[ -n "${LOAD_PATH:-}" ]]; then
  EXTRA+=(--load-path "${LOAD_PATH}")
fi

echo "Launching middle-ppl-doc: nproc_per_node=${NPROC_PER_NODE} name=${NAME}" >&2

if [[ "${NPROC_PER_NODE}" -eq 1 ]] && [[ -z "${WORLD_SIZE:-}" ]]; then
  exec python "${TRAINER}" "${COMMON_ARGS[@]}" "${EXTRA[@]}"
fi

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${TRAINER}" \
  "${COMMON_ARGS[@]}" \
  "${EXTRA[@]}"
