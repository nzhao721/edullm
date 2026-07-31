#!/usr/bin/env bash
# Launch one Skill-It OLMo2-370M arm (probe or derivative A).
#
# Required:
#   ARM_ID          skillit-probe | skillit-deriv (or any id)
#   A_MODE          probe | derivative
#   SAVE_FOLDER     local checkpoint dir
#   PROGRESS_DIR    local progress dir
#
# Data (edullm-data → local stage; no legacy edullm-datasets / pre-existing pool):
#   POOL_DIR        staging root (default: sibling of PROGRESS_DIR named pool/)
#   DATASET_ID      published id (default: pretrain/olmo-original-30b)
#   DATASET_VERSION optional pin (default: resolve_latest)
#   ARM_WEIGHTS_JSON  per-arm sidecar from prepare_skillit_370m_data.py (recommended)
#
# Optional:
#   NPROC           1 → python; >1 → torchrun (default 1)
#   DEVICE_BATCH_SIZE  default 32
#   SEED            default 42
#   LR_ALPHA_F      default 1.0
#   A_OFFLINE       path to A_offline.npy (default: artifacts/probes_full or artifacts/)
#   MIXLAW_FIT_JSON path to mixlaw_fit_chinchilla.json
#   S3_EXPORT       0 → --no-s3-export (local smoke only; default: fail-closed S3 on)
#   WANDB_PROJECT   default skillit
#   WANDB_MODE      online|offline|disabled (default: online if WANDB_API_KEY set else disabled)
#   WANDB_ENTITY / WANDB_RUN_NAME / WANDB_GROUP / WANDB_UPLOAD_EXISTING
#   EXTRA_ARGS      extra trainer flags
#   PYTHON / TRAIN_SCRIPT
#
# Note: missing ``wandb`` in the Python env only disables W&B soft-log; it does
# not turn off fail-closed S3 export to s3://edullm-checkpoints/skillit/.
#
# Examples:
#   NPROC=1 ARM_ID=skillit-probe A_MODE=probe \
#     POOL_DIR=$WORK/pool SAVE_FOLDER=... PROGRESS_DIR=... \
#     bash launch_arm.sh
#
#   NPROC=8 ARM_ID=skillit-deriv A_MODE=derivative S3_EXPORT=0 ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer an explicit checkout (FarmShare ephemeral RUN_DIR copies launch into scratch).
if [[ -n "${EDULLM_ROOT:-}" && -d "${EDULLM_ROOT}/experiments/curriculum" ]]; then
  REPO_ROOT="$(cd "${EDULLM_ROOT}" && pwd)"
elif [[ -d "${SCRIPT_DIR}/../../../experiments/curriculum" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
else
  REPO_ROOT=""
fi
MIXLAW_ROOT="${MIXLAW_ROOT:-${REPO_ROOT:+${REPO_ROOT}/experiments/skill-dag/mixlaw}}"
if [[ -z "${MIXLAW_ROOT}" || ! -d "${MIXLAW_ROOT}" ]]; then
  MIXLAW_ROOT="$(cd "${SCRIPT_DIR}/../mixlaw" 2>/dev/null && pwd || true)"
fi
# curriculum (train_curriculum_regmix_370m) + token-selection + mixlaw domain_stream
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${MIXLAW_ROOT}" && -d "${MIXLAW_ROOT}" ]]; then
  export PYTHONPATH="${MIXLAW_ROOT}:${PYTHONPATH}"
fi
if [[ -n "${REPO_ROOT}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/experiments/curriculum:${REPO_ROOT}/experiments/token-selection:${PYTHONPATH}"
fi

TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_skillit_370m.py}"
PYTHON="${PYTHON:-python}"
NPROC="${NPROC:-1}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
SEED="${SEED:-42}"
LR_ALPHA_F="${LR_ALPHA_F:-1.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ -z "${A_OFFLINE:-}" ]]; then
  # Plan default: artifacts/probes_full/A_offline.npy (legacy artifacts/ fallback).
  if [[ -f "${SCRIPT_DIR}/artifacts/probes_full/A_offline.npy" ]]; then
    A_OFFLINE="${SCRIPT_DIR}/artifacts/probes_full/A_offline.npy"
  elif [[ -n "${REPO_ROOT}" && -f "${REPO_ROOT}/experiments/skill-dag/skillit/artifacts/probes_full/A_offline.npy" ]]; then
    A_OFFLINE="${REPO_ROOT}/experiments/skill-dag/skillit/artifacts/probes_full/A_offline.npy"
  else
    A_OFFLINE="${SCRIPT_DIR}/artifacts/A_offline.npy"
  fi
fi
if [[ -z "${MIXLAW_FIT_JSON:-}" ]]; then
  if [[ -n "${MIXLAW_ROOT}" && -f "${MIXLAW_ROOT}/mixlaw_fit_chinchilla.json" ]]; then
    MIXLAW_FIT_JSON="${MIXLAW_ROOT}/mixlaw_fit_chinchilla.json"
  else
    MIXLAW_FIT_JSON="${SCRIPT_DIR}/../mixlaw/mixlaw_fit_chinchilla.json"
  fi
fi
DATASET_ID="${DATASET_ID:-pretrain/olmo-original-30b}"
WANDB_PROJECT="${WANDB_PROJECT:-skillit}"
if [[ -z "${WANDB_MODE:-}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_MODE="online"
  else
    WANDB_MODE="disabled"
  fi
fi

: "${ARM_ID:?ARM_ID is required}"
: "${A_MODE:?A_MODE is required (probe|derivative)}"
: "${SAVE_FOLDER:?SAVE_FOLDER is required}"
: "${PROGRESS_DIR:?PROGRESS_DIR is required}"

if [[ -z "${REPO_ROOT}" || ! -d "${REPO_ROOT}/experiments/curriculum" ]]; then
  echo "[launch_arm] error: cannot resolve EDULLM_ROOT/REPO_ROOT with experiments/curriculum" >&2
  echo "  set EDULLM_ROOT to the edullm checkout (submit_skillit_370m.sh sets this)." >&2
  exit 2
fi
if [[ ! -f "${REPO_ROOT}/experiments/curriculum/train_curriculum_regmix_370m.py" ]]; then
  echo "[launch_arm] error: missing ${REPO_ROOT}/experiments/curriculum/train_curriculum_regmix_370m.py" >&2
  exit 2
fi

POOL_DIR="${POOL_DIR:-$(cd "$(dirname "${PROGRESS_DIR}")" && pwd)/pool}"

ARGS=(
  --name "${ARM_ID}"
  --arm-id "${ARM_ID}"
  --a-mode "${A_MODE}"
  --pool-dir "${POOL_DIR}"
  --dataset-id "${DATASET_ID}"
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --device-batch-size "${DEVICE_BATCH_SIZE}"
  --seed "${SEED}"
  --lr-alpha-f "${LR_ALPHA_F}"
  --a-offline "${A_OFFLINE}"
  --mixlaw-fit-json "${MIXLAW_FIT_JSON}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode "${WANDB_MODE}"
)

if [[ -n "${DATASET_VERSION:-}" ]]; then
  ARGS+=(--dataset-version "${DATASET_VERSION}")
fi

if [[ -n "${ARM_WEIGHTS_JSON:-}" ]]; then
  ARGS+=(--arm-weights-json "${ARM_WEIGHTS_JSON}")
fi

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
  ARGS+=(--wandb-run-name "${WANDB_RUN_NAME}")
else
  ARGS+=(--wandb-run-name "${ARM_ID}")
fi
if [[ -n "${WANDB_GROUP:-}" ]]; then
  ARGS+=(--wandb-group "${WANDB_GROUP}")
fi
if [[ "${WANDB_UPLOAD_EXISTING:-0}" == "1" ]]; then
  ARGS+=(--wandb-upload-existing)
fi

if [[ "${S3_EXPORT:-1}" == "0" ]]; then
  ARGS+=(--no-s3-export)
else
  ARGS+=(--s3-export)
fi

# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS} )

if [[ "${WANDB_MODE}" != "disabled" ]]; then
  if ! "${PYTHON}" -c "import wandb" >/dev/null 2>&1; then
    echo "[launch_arm] warn: wandb not installed in PYTHON=${PYTHON}; W&B soft-disabled, S3 export unchanged" >&2
  fi
fi

echo "[launch_arm] ARM_ID=${ARM_ID} A_MODE=${A_MODE} NPROC=${NPROC} DATASET_ID=${DATASET_ID} POOL_DIR=${POOL_DIR} S3_EXPORT=${S3_EXPORT:-1} WANDB_PROJECT=${WANDB_PROJECT} WANDB_MODE=${WANDB_MODE}"

if [[ "${NPROC}" -le 1 ]]; then
  exec "${PYTHON}" "${TRAIN_SCRIPT}" "${ARGS[@]}" "${EXTRA[@]}"
else
  exec torchrun --standalone --nproc_per_node="${NPROC}" \
    "${TRAIN_SCRIPT}" "${ARGS[@]}" "${EXTRA[@]}"
fi
