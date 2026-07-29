#!/usr/bin/env bash
# Launch Control (plain CE on RegMix 10B) on 1..N GPUs.
#
# Required env / args (no host-specific defaults):
#   TRAIN_PATHS_FILE  — memmap path list from prepare_control_data.py
#   SAVE_FOLDER       — checkpoint root
#   PROGRESS_DIR      — metrics / run_meta
#
# Optional:
#   NPROC             — processes per node (default: 1, or WORLD_SIZE if set)
#   NAME              — run id (default: control-regmix10b-v1)
#   LENGTH_TOKENS     — default 10000000000 → 2384 steps
#   TASK_LOSS_EVAL    — 0 to disable post-save evals (default: enabled by trainer)
#   S3_EXPORT         — 0 to disable post-save S3 sync (default: on when aws CLI present)
#   EXTRA_ARGS        — extra flags forwarded to the trainer
#
# Examples:
#   # 1 GPU
#   TRAIN_PATHS_FILE=.../paths_train.txt SAVE_FOLDER=.../ckpts PROGRESS_DIR=.../progress \
#     NPROC=1 ./launch_control.sh
#
#   # Multi-GPU (world size = NPROC; discovered by torchrun — no hardcoded device IDs)
#   TRAIN_PATHS_FILE=... SAVE_FOLDER=... PROGRESS_DIR=... NPROC=4 ./launch_control.sh
#
#   # Or call torchrun yourself:
#   torchrun --standalone --nproc_per_node=4 train_ce_regmix_olmo_370m.py \
#     --train-paths-file ... --save-folder ... --progress-dir ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/experiments/token-selection${PYTHONPATH:+:${PYTHONPATH}}"

: "${TRAIN_PATHS_FILE:?Set TRAIN_PATHS_FILE to the RegMix paths list}"
: "${SAVE_FOLDER:?Set SAVE_FOLDER to the checkpoint directory}"
: "${PROGRESS_DIR:?Set PROGRESS_DIR to the progress/metrics directory}"

NAME="${NAME:-control-regmix10b-v1}"
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

TRAINER="${SCRIPT_DIR}/train_ce_regmix_olmo_370m.py"
COMMON_ARGS=(
  --name "${NAME}"
  --train-paths-file "${TRAIN_PATHS_FILE}"
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --length-tokens "${LENGTH_TOKENS}"
)

# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS:-} )

echo "Launching control: nproc_per_node=${NPROC_PER_NODE} name=${NAME}" >&2

if [[ "${NPROC_PER_NODE}" -eq 1 ]] && [[ -z "${WORLD_SIZE:-}" ]]; then
  # Single process without torchrun is fine; trainer still prepares the env.
  exec python "${TRAINER}" "${COMMON_ARGS[@]}" "${EXTRA[@]}"
fi

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${TRAINER}" \
  "${COMMON_ARGS[@]}" \
  "${EXTRA[@]}"
