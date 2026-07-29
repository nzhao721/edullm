#!/usr/bin/env bash
# Launch one curriculum / control arm. AWS-agnostic: caller sets paths + NPROC.
#
# Required env (or flags):
#   ARM_ID              e.g. control | linear10-cr | expand-flesch | ...
#   PACING              control | linear_n10 | expanding_25_1000 | warmup_1000 | interleave_i10_linear
#   DIFFICULTY_METRIC   compression_ratio | flesch | mtld | learnability  (omit for control)
#   SAVE_FOLDER         local checkpoint dir
#   PROGRESS_DIR        local progress dir
#   TRAIN_PATHS_FILE    control only: list of flat memmap paths
#   CURRICULUM_INDEX    curriculum arms: local curriculum/ root
#
# Optional:
#   NPROC               default 1 → plain python; >1 → torchrun --nproc_per_node=NPROC
#   DEVICE_BATCH_SIZE   sequences per microbatch (default 32)
#   SEED                default 42
#   LR_ALPHA_F          default 1.0 (constant LR after warmup)
#   EXTRA_ARGS          extra trainer CLI flags
#   PYTHON              python executable (default: python)
#   TRAIN_SCRIPT        override path to train_curriculum_regmix_370m.py
#
# Example (1 rank):
#   ARM_ID=control PACING=control TRAIN_PATHS_FILE=/data/paths.txt \
#     SAVE_FOLDER=/scratch/control/ckpts PROGRESS_DIR=/scratch/control/progress \
#     bash launch_arm.sh
#
# Example (N ranks):
#   NPROC=4 ARM_ID=linear10-cr PACING=linear_n10 DIFFICULTY_METRIC=compression_ratio \
#     CURRICULUM_INDEX=/data/curriculum \
#     SAVE_FOLDER=/scratch/linear10-cr/ckpts PROGRESS_DIR=/scratch/linear10-cr/progress \
#     bash launch_arm.sh
#
# Optional post-hoc EMA (after training; mirrors s3://edullm-checkpoints/curriculum/<arm_id>/checkpoints):
#   python ../ema_merge_checkpoints.py \
#     --checkpoints-root "$SAVE_FOLDER" --arm-id "$ARM_ID"
#   # --task-loss is default on; pass --no-task-loss to skip the 20-label eval

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${CUR_ROOT}/train_curriculum_regmix_370m.py}"
PYTHON="${PYTHON:-python}"
NPROC="${NPROC:-1}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
SEED="${SEED:-42}"
LR_ALPHA_F="${LR_ALPHA_F:-1.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

: "${ARM_ID:?ARM_ID is required}"
: "${PACING:?PACING is required}"
: "${SAVE_FOLDER:?SAVE_FOLDER is required}"
: "${PROGRESS_DIR:?PROGRESS_DIR is required}"

ARGS=(
  --name "${ARM_ID}"
  --arm-id "${ARM_ID}"
  --pacing "${PACING}"
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --device-batch-size "${DEVICE_BATCH_SIZE}"
  --seed "${SEED}"
  --lr-alpha-f "${LR_ALPHA_F}"
)

if [[ "${PACING}" == "control" ]]; then
  : "${TRAIN_PATHS_FILE:?TRAIN_PATHS_FILE is required for control}"
  ARGS+=(--train-paths-file "${TRAIN_PATHS_FILE}")
else
  : "${DIFFICULTY_METRIC:?DIFFICULTY_METRIC is required for curriculum pacing}"
  : "${CURRICULUM_INDEX:?CURRICULUM_INDEX is required for curriculum pacing}"
  ARGS+=(--difficulty-metric "${DIFFICULTY_METRIC}" --curriculum-index "${CURRICULUM_INDEX}")
fi

# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS} )

echo "[launch_arm] ARM_ID=${ARM_ID} PACING=${PACING} NPROC=${NPROC} DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE}"

if [[ "${NPROC}" -le 1 ]]; then
  exec "${PYTHON}" "${TRAIN_SCRIPT}" "${ARGS[@]}" "${EXTRA[@]}"
else
  exec torchrun --standalone --nproc_per_node="${NPROC}" \
    "${TRAIN_SCRIPT}" "${ARGS[@]}" "${EXTRA[@]}"
fi
