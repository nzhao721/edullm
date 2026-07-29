#!/usr/bin/env bash
# Launch learnability-doc CE train (1..N GPUs via torchrun / WORLD_SIZE).
# Does not hardcode GPU count, device IDs, or host paths.
#
# Required:
#   TRAIN_PATHS_FILE  corpus paths_train.txt
#   SAVE_FOLDER       permanent checkpoints
#   PROGRESS_DIR      run meta / progress
#
# Optional:
#   NPROC / WORLD_SIZE   default 1
#   NAME                 default edullm-370M-learnability-doc-10b
#   LENGTH_TOKENS        default 10000000000 (~2384 steps)
#   TASK_LOSS_EVAL       0 to disable post-save evals
#   TASK_LOSS_EVAL_SCRIPT / TASK_LOSS_OUT_DIR
#   FRESH=1, LOAD_PATH, EXTRA_ARGS
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${TRAIN_PATHS_FILE:?Set TRAIN_PATHS_FILE to corpus paths_train.txt}"
: "${SAVE_FOLDER:?Set SAVE_FOLDER for permanent checkpoints}"
: "${PROGRESS_DIR:?Set PROGRESS_DIR for run meta / progress}"

NAME="${NAME:-edullm-370M-learnability-doc-10b}"
LENGTH_TOKENS="${LENGTH_TOKENS:-10000000000}"
TASK_LOSS_RESULTS_DIR="${TASK_LOSS_RESULTS_DIR:-${TASK_LOSS_OUT_DIR:-${TS_ROOT}/task_loss_results/learnability-doc}}"

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

export PYTHONPATH="${TS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

EXTRA=()
if [[ "${FRESH:-0}" == "1" ]]; then
  EXTRA+=(--fresh)
fi
if [[ "${NO_TASK_LOSS:-0}" == "1" ]] || [[ "${TASK_LOSS_EVAL:-1}" == "0" ]]; then
  EXTRA+=(--no-task-loss-on-save)
fi
if [[ -n "${LOAD_PATH:-}" ]]; then
  EXTRA+=(--load-path "${LOAD_PATH}")
fi
if [[ -n "${TASK_LOSS_EVAL_SCRIPT:-}" ]]; then
  EXTRA+=(--task-loss-eval-script "${TASK_LOSS_EVAL_SCRIPT}")
fi
# shellcheck disable=SC2206
EXTRA+=( ${EXTRA_ARGS:-} )

COMMON_ARGS=(
  --name "${NAME}"
  --train-paths-file "${TRAIN_PATHS_FILE}"
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --length-tokens "${LENGTH_TOKENS}"
  --task-loss-results-dir "${TASK_LOSS_RESULTS_DIR}"
)

TRAINER="${SCRIPT_DIR}/train_ce_learnability_doc_olmo_370m.py"
echo "Launching learnability-doc: nproc_per_node=${NPROC_PER_NODE} name=${NAME}" >&2

if [[ "${NPROC_PER_NODE}" -eq 1 ]] && [[ -z "${WORLD_SIZE:-}" ]]; then
  exec python "${TRAINER}" "${COMMON_ARGS[@]}" "${EXTRA[@]}"
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${TRAINER}" \
  "${COMMON_ARGS[@]}" \
  "${EXTRA[@]}"
