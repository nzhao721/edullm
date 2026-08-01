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
#   DATASET_ID      pinned published id (pretrain/olmo-127b)
#   DATASET_VERSION pinned published version (v1)
#   ARM_WEIGHTS_JSON  per-arm sidecar from prepare_skillit_370m_data.py (recommended)
#
# Optional:
#   NPROC           1 → python; >1 → torchrun (default 1)
#   RESUME_MODE     required: fresh | resume
#   LOAD_PATH       required for resume; local step dir or legacy S3 bootstrap step
#   DEVICE_BATCH_SIZE  default 32
#   DEVICE_EVAL_BATCH_SIZE per-rank eval batch (default: 32/NPROC)
#   LADDER_BASE_CONFIG compatible OLMES evaluation YAML (required)
#   TASK_LOSS_EVAL_SCRIPT shared all-rank evaluator (repo default)
#   SEED            default 42
#   LR_ALPHA_F      default 1.0
#   A_OFFLINE       path to A_offline.npy (default: artifacts/probes_full or artifacts/)
#   MIXLAW_FIT_JSON path to mixlaw_fit_chinchilla.json
#   WANDB_PROJECT   default skillit
#   WANDB_MODE      production requires online
#   ALLOW_LOCAL_ONLY 1 permits offline/disabled W&B for local smoke only
#   WANDB_ENTITY / WANDB_RUN_NAME / WANDB_GROUP / WANDB_UPLOAD_EXISTING
#   EXTRA_ARGS      extra trainer flags
#   PYTHON / TRAIN_SCRIPT
#
# Examples:
#   NPROC=1 ARM_ID=skillit-probe A_MODE=probe \
#     POOL_DIR=$WORK/pool SAVE_FOLDER=... PROGRESS_DIR=... \
#     bash launch_arm.sh
#
#   NPROC=8 ARM_ID=skillit-deriv A_MODE=derivative ALLOW_LOCAL_ONLY=1 ...

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
DATASET_ID="${DATASET_ID:-pretrain/olmo-127b}"
DATASET_VERSION="${DATASET_VERSION:-v1}"
PINNED_DATASET_ID="pretrain/olmo-127b"
PINNED_DATASET_VERSION="v1"
RESUME_MODE="${RESUME_MODE:-}"
LOAD_PATH="${LOAD_PATH:-}"
TASK_LOSS_ON_SAVE="${TASK_LOSS_ON_SAVE:-1}"
TASK_LOSS_EVAL_SCRIPT="${TASK_LOSS_EVAL_SCRIPT:-${REPO_ROOT}/scripts/farmshare/task_loss/eval_task_loss_olmo_core.py}"
LADDER_BASE_CONFIG="${LADDER_BASE_CONFIG:-}"
WANDB_PROJECT="${WANDB_PROJECT:-skillit}"
ALLOW_LOCAL_ONLY="${ALLOW_LOCAL_ONLY:-0}"
if [[ -z "${WANDB_MODE:-}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_MODE="online"
  else
    WANDB_MODE="disabled"
  fi
fi
if [[ "${ALLOW_LOCAL_ONLY}" != "1" && "${WANDB_MODE}" != "online" ]]; then
  echo "[launch_arm] production requires WANDB_MODE=online; set ALLOW_LOCAL_ONLY=1 only for local smoke" >&2
  exit 2
fi

: "${ARM_ID:?ARM_ID is required}"
: "${A_MODE:?A_MODE is required (probe|derivative)}"
: "${SAVE_FOLDER:?SAVE_FOLDER is required}"
: "${PROGRESS_DIR:?PROGRESS_DIR is required}"

if [[ "${DATASET_ID}" != "${PINNED_DATASET_ID}" || "${DATASET_VERSION}" != "${PINNED_DATASET_VERSION}" ]]; then
  echo "[launch_arm] SkillIt source is pinned to ${PINNED_DATASET_ID}/${PINNED_DATASET_VERSION}; got ${DATASET_ID}/${DATASET_VERSION}" >&2
  exit 2
fi
case "${RESUME_MODE}" in
  fresh)
    if [[ -n "${LOAD_PATH}" ]]; then
      echo "[launch_arm] RESUME_MODE=fresh conflicts with LOAD_PATH" >&2
      exit 2
    fi
    ;;
  resume)
    if [[ -z "${LOAD_PATH}" ]]; then
      echo "[launch_arm] RESUME_MODE=resume requires LOAD_PATH" >&2
      exit 2
    fi
    ;;
  *)
    echo "[launch_arm] set RESUME_MODE=fresh or RESUME_MODE=resume" >&2
    exit 2
    ;;
esac

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

if ! [[ "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[launch_arm] NPROC must be a positive integer" >&2
  exit 2
fi
if (( 32 % NPROC != 0 )); then
  echo "[launch_arm] NPROC=${NPROC} must divide aggregate eval batch 32" >&2
  exit 2
fi
DEVICE_EVAL_BATCH_SIZE="${DEVICE_EVAL_BATCH_SIZE:-$((32 / NPROC))}"
TASK_LOSS_NPROC="${TASK_LOSS_NPROC:-${NPROC}}"
if [[ "${TASK_LOSS_NPROC}" != "${NPROC}" ]]; then
  echo "[launch_arm] in-process all-rank eval requires TASK_LOSS_NPROC=NPROC (${NPROC})" >&2
  exit 2
fi
export TASK_LOSS_NPROC

if [[ -f "${POOL_DIR}/edullm_data_source.json" || -f "${POOL_DIR}/_EDULLM_DATA_SOURCE.json" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/prepare_skillit_370m_data.py" \
    --pool-dir "${POOL_DIR}" \
    --dataset-id "${DATASET_ID}" \
    --dataset-version "${DATASET_VERSION}" \
    --validate-pool-only
else
  echo "[launch_arm] no staged pool marker; rank 0 will stage pinned ${DATASET_ID}/${DATASET_VERSION}" >&2
fi

if [[ "${TASK_LOSS_ON_SAVE}" != "0" ]]; then
  if [[ -z "${LADDER_BASE_CONFIG}" || ! -f "${LADDER_BASE_CONFIG}" ]]; then
    echo "[launch_arm] LADDER_BASE_CONFIG must name an existing compatible eval YAML" >&2
    exit 2
  fi
  if [[ ! -f "${TASK_LOSS_EVAL_SCRIPT}" ]]; then
    echo "[launch_arm] missing shared evaluator ${TASK_LOSS_EVAL_SCRIPT}" >&2
    exit 2
  fi
  if [[ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
    echo "[launch_arm] HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is required for fail-closed checkpoint eval" >&2
    exit 2
  fi
  export LADDER_BASE_CONFIG TASK_LOSS_EVAL_SCRIPT
  "${PYTHON}" - <<'PY'
from olmo.eval.downstream import label_to_task_map
from mixlaw_common import LADDER_TASK_LOSS_LABELS
missing = [label for label in LADDER_TASK_LOSS_LABELS if label not in label_to_task_map]
if missing:
    raise SystemExit(f"installed ai2-olmo lacks required OLMES bpb labels: {missing}")
print(f"[launch_arm] OLMES preflight ok ({len(LADDER_TASK_LOSS_LABELS)} labels)")
PY
fi

ARGS=(
  --name "${ARM_ID}"
  --arm-id "${ARM_ID}"
  --a-mode "${A_MODE}"
  --pool-dir "${POOL_DIR}"
  --dataset-id "${DATASET_ID}"
  --dataset-version "${DATASET_VERSION}"
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --device-batch-size "${DEVICE_BATCH_SIZE}"
  --seed "${SEED}"
  --lr-alpha-f "${LR_ALPHA_F}"
  --device-eval-batch-size "${DEVICE_EVAL_BATCH_SIZE}"
  --task-loss-eval-script "${TASK_LOSS_EVAL_SCRIPT}"
  --a-offline "${A_OFFLINE}"
  --mixlaw-fit-json "${MIXLAW_FIT_JSON}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode "${WANDB_MODE}"
)

if [[ -n "${ARM_WEIGHTS_JSON:-}" ]]; then
  ARGS+=(--arm-weights-json "${ARM_WEIGHTS_JSON}")
fi

if [[ "${RESUME_MODE}" == "fresh" ]]; then
  ARGS+=(--fresh)
else
  ARGS+=(--load-path "${LOAD_PATH}")
fi
if [[ "${TASK_LOSS_ON_SAVE}" == "0" ]]; then
  ARGS+=(--no-task-loss-on-save)
else
  ARGS+=(--task-loss-on-save)
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
if [[ "${ALLOW_LOCAL_ONLY}" == "1" ]]; then
  ARGS+=(--allow-local-only)
fi

# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS} )

if [[ "${WANDB_MODE}" == "online" ]]; then
  test -n "${WANDB_API_KEY:-}" || {
    echo "[launch_arm] WANDB_API_KEY is required for production online runs" >&2
    exit 2
  }
  if ! "${PYTHON}" -c "import wandb" >/dev/null 2>&1; then
    echo "[launch_arm] wandb is required for production checkpoint durability" >&2
    exit 2
  fi
fi

echo "[launch_arm] ARM_ID=${ARM_ID} A_MODE=${A_MODE} NPROC=${NPROC} RESUME_MODE=${RESUME_MODE} DATASET=${DATASET_ID}/${DATASET_VERSION} POOL_DIR=${POOL_DIR} TASK_LOSS_NPROC=${TASK_LOSS_NPROC} DEVICE_EVAL_BATCH_SIZE=${DEVICE_EVAL_BATCH_SIZE} WANDB_PROJECT=${WANDB_PROJECT} WANDB_MODE=${WANDB_MODE} ALLOW_LOCAL_ONLY=${ALLOW_LOCAL_ONLY}"

if [[ "${NPROC}" -le 1 ]]; then
  exec "${PYTHON}" "${TRAIN_SCRIPT}" "${ARGS[@]}" "${EXTRA[@]}"
else
  exec torchrun --standalone --nproc_per_node="${NPROC}" \
    "${TRAIN_SCRIPT}" "${ARGS[@]}" "${EXTRA[@]}"
fi
