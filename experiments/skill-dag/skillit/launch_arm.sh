#!/usr/bin/env bash
# Launch one Skill-It OLMo2-370M arm (probe or derivative A).
#
# Required:
#   ARM_ID          skillit-probe | skillit-deriv (or any id)
#   A_MODE          probe | derivative
#   SAVE_FOLDER     local checkpoint dir
#   PROGRESS_DIR    local progress dir
#   POOL_DIR        olmohq working pool root (tokenized/<domain>/<domain>.npy)
#
# Optional:
#   NPROC           1 → python; >1 → torchrun (default 1)
#   DEVICE_BATCH_SIZE  default 32
#   SEED            default 42
#   LR_ALPHA_F      default 1.0
#   A_OFFLINE       path to A_offline.npy (default: skillit/artifacts/A_offline.npy)
#   MIXLAW_FIT_JSON path to mixlaw_fit_chinchilla.json
#   S3_EXPORT       0 to disable live aws s3 sync (default: enabled if aws present)
#   EXTRA_ARGS      extra trainer flags
#   PYTHON / TRAIN_SCRIPT
#
# Examples:
#   NPROC=1 ARM_ID=skillit-probe A_MODE=probe \
#     POOL_DIR=$WORK/olmohq-pool SAVE_FOLDER=... PROGRESS_DIR=... \
#     bash launch_arm.sh
#
#   NPROC=8 ARM_ID=skillit-deriv A_MODE=derivative S3_EXPORT=0 ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_skillit_370m.py}"
PYTHON="${PYTHON:-python}"
NPROC="${NPROC:-1}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
SEED="${SEED:-42}"
LR_ALPHA_F="${LR_ALPHA_F:-1.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
A_OFFLINE="${A_OFFLINE:-${SCRIPT_DIR}/artifacts/A_offline.npy}"
MIXLAW_FIT_JSON="${MIXLAW_FIT_JSON:-${SCRIPT_DIR}/../mixlaw/mixlaw_fit_chinchilla.json}"

: "${ARM_ID:?ARM_ID is required}"
: "${A_MODE:?A_MODE is required (probe|derivative)}"
: "${SAVE_FOLDER:?SAVE_FOLDER is required}"
: "${PROGRESS_DIR:?PROGRESS_DIR is required}"
: "${POOL_DIR:?POOL_DIR is required}"

ARGS=(
  --name "${ARM_ID}"
  --arm-id "${ARM_ID}"
  --a-mode "${A_MODE}"
  --pool-dir "${POOL_DIR}"
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --device-batch-size "${DEVICE_BATCH_SIZE}"
  --seed "${SEED}"
  --lr-alpha-f "${LR_ALPHA_F}"
  --a-offline "${A_OFFLINE}"
  --mixlaw-fit-json "${MIXLAW_FIT_JSON}"
)

# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS} )

echo "[launch_arm] ARM_ID=${ARM_ID} A_MODE=${A_MODE} NPROC=${NPROC} POOL_DIR=${POOL_DIR}"

if [[ "${NPROC}" -le 1 ]]; then
  exec "${PYTHON}" "${TRAIN_SCRIPT}" "${ARGS[@]}" "${EXTRA[@]}"
else
  exec torchrun --standalone --nproc_per_node="${NPROC}" \
    "${TRAIN_SCRIPT}" "${ARGS[@]}" "${EXTRA[@]}"
fi
