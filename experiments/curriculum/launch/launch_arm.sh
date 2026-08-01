#!/usr/bin/env bash
# Launch one curriculum / control arm on an ephemeral runtime.
#
# Scratch is assumed empty at job start and wiped after. Stage train/curriculum
# bytes from s3://edullm-data/ into a job-scoped cache. All run artifacts remain
# on scratch and upload to W&B project "curriculum". Production checkpoint
# uploads are synchronous and fail-closed. Nothing from the run is written to S3.
#
# Required env:
#   ARM_ID              e.g. control | linear10-cr | expand-flesch | ...
#   PACING              control | linear_n10 | expanding_25_1000 | warmup_1000 | interleave_i10_linear
#   DIFFICULTY_METRIC   compression_ratio | flesch | mtld | learnability  (omit for control)
#   SAVE_FOLDER         job-scoped local checkpoint dir (not durable storage)
#   PROGRESS_DIR        job-scoped local progress dir (not durable storage)
#
# Data (defaults fetch from s3://edullm-data/ on a clean machine):
#   TRAIN_DATASET_ID           default pretrain/regmix-10b
#   TRAIN_DATASET_VERSION      optional pin (else resolve_latest)
#   CURRICULUM_DATASET_ID      curriculum arms: default curriculum/regmix-370m
#   CURRICULUM_DATASET_VERSION optional pin
#   CURRICULUM_ORDER_GROUP     optional override (default: from DIFFICULTY_METRIC → compression|flesch|mtld|learnability)
#   DATA_CACHE_DIR / EDULLM_DATA_CACHE  job-scoped staging root for fetch-if-missing
#   TRAIN_PATHS_FILE           optional: paths already staged in THIS job
#   CURRICULUM_INDEX           rejected (legacy document-local coordinates)
#
# W&B (the sole production artifact backend):
#   RUN_DIR             job scratch root; sources ${RUN_DIR}/wandb-session.env + aws-session.env
#   WANDB_PROJECT       default: curriculum
#   WANDB_ENTITY        optional (empty = account default; do not invent)
#   WANDB_RUN_NAME      default: ARM_ID (or ${ARM_ID}-${SLURM_JOB_ID} when set)
#   WANDB_MODE          default: online (requires wandb-session.env); local smoke: disabled
#   WANDB_UPLOAD_EXISTING=1  upload existing local ckpts/evals on start
#   ALLOW_LOCAL_ONLY=1  scratch-only local-smoke escape hatch (with WANDB_MODE=disabled)
#
# FarmShare enablement (laptop → scratch, same as SmolLM):
#   bash scripts/farmshare/push_aws_session_to_farmshare.sh "$RUN_DIR"
#   bash scripts/farmshare/push_wandb_session_to_farmshare.sh "$RUN_DIR"
#
# Optional:
#   NPROC               default 1 → plain python; >1 → torchrun --nproc_per_node=NPROC
#   DEVICE_BATCH_SIZE   sequences per microbatch (default 32)
#   SEED                default 42
#   LR_ALPHA_F          default 1.0 (constant LR after warmup)
#   FRESH=1             explicit start-from-scratch mode (required unless LOAD_PATH is set)
#   LOAD_PATH           local step dir or wandb-artifact://entity/project/name:version
#   LADDER_BASE_CONFIG  required config YAML when task-loss eval is enabled
#   HF_SESSION_ENV      optional hf-session.env; HF_TOKEN is preflighted without network
#   TASK_LOSS_OUT_DIR   default: $PROGRESS_DIR/task_loss_results
#   METRICS_DIR         default: sibling of PROGRESS_DIR named metrics
#   EXTRA_ARGS          extra trainer CLI flags
#   PYTHON              python executable (default: python)
#   TRAIN_SCRIPT        override path to train_curriculum_regmix_370m.py
#
# Example (1 rank, empty job scratch — stages pretrain/regmix-10b if missing):
#   RUN_DIR=/tmp/job-$SLURM_JOB_ID
#   mkdir -p "$RUN_DIR"/{ckpts,progress,cache}
#   bash scripts/farmshare/push_wandb_session_to_farmshare.sh "$RUN_DIR"
#   bash scripts/farmshare/push_aws_session_to_farmshare.sh "$RUN_DIR"
#   ARM_ID=control PACING=control \
#     FRESH=1 LADDER_BASE_CONFIG=/path/to/ladder-config.yaml \
#     SAVE_FOLDER=$RUN_DIR/ckpts PROGRESS_DIR=$RUN_DIR/progress \
#     DATA_CACHE_DIR=$RUN_DIR/cache \
#     bash launch_arm.sh
#
# Example (N ranks, curriculum — requires published curriculum/* on edullm-data):
#   NPROC=4 ARM_ID=linear10-cr PACING=linear_n10 DIFFICULTY_METRIC=compression_ratio \
#     FRESH=1 LADDER_BASE_CONFIG=/path/to/ladder-config.yaml \
#     SAVE_FOLDER=$RUN_DIR/ckpts PROGRESS_DIR=$RUN_DIR/progress \
#     DATA_CACHE_DIR=$RUN_DIR/cache \
#     bash launch_arm.sh
#
# Local smoke (no durable sink):
#   FRESH=1 TASK_LOSS_EVAL=0 WANDB_MODE=disabled ALLOW_LOCAL_ONLY=1 bash launch_arm.sh
#
# Resume (download a W&B model artifact into SAVE_FOLDER, or pass a local staged dir):
#   LOAD_PATH=wandb-artifact://entity/curriculum/control-checkpoint-step0001250:v0 \
#     bash launch_arm.sh
#
# Optional post-hoc EMA (using scratch checkpoints or W&B artifact downloads):
#   python ../ema_merge_checkpoints.py \
#     --checkpoints-root "$SAVE_FOLDER" --arm-id "$ARM_ID"

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
TRAIN_DATASET_ID="${TRAIN_DATASET_ID:-pretrain/regmix-10b}"
TASK_LOSS_EVAL="${TASK_LOSS_EVAL:-1}"

: "${ARM_ID:?ARM_ID is required}"
: "${PACING:?PACING is required}"
: "${SAVE_FOLDER:?SAVE_FOLDER is required (job-scoped scratch)}"
: "${PROGRESS_DIR:?PROGRESS_DIR is required (job-scoped scratch)}"

mkdir -p "${SAVE_FOLDER}" "${PROGRESS_DIR}"
TASK_LOSS_OUT_DIR="${TASK_LOSS_OUT_DIR:-${PROGRESS_DIR}/task_loss_results}"
METRICS_DIR="${METRICS_DIR:-$(dirname "${PROGRESS_DIR}")/metrics}"
mkdir -p "${METRICS_DIR}" "${PROGRESS_DIR}/../wandb"

# --- W&B / AWS session (SmolLM FarmShare protocol) ---
WANDB_PROJECT="${WANDB_PROJECT:-curriculum}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  WANDB_RUN_NAME="${WANDB_RUN_NAME:-${ARM_ID}-${SLURM_JOB_ID}}"
else
  WANDB_RUN_NAME="${WANDB_RUN_NAME:-${ARM_ID}}"
fi
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_UPLOAD_EXISTING="${WANDB_UPLOAD_EXISTING:-0}"
ALLOW_LOCAL_ONLY="${ALLOW_LOCAL_ONLY:-0}"

WANDB_SESSION_ENV="${WANDB_SESSION_ENV:-}"
if [[ -z "${WANDB_SESSION_ENV}" && -n "${RUN_DIR:-}" && -f "${RUN_DIR}/wandb-session.env" ]]; then
  WANDB_SESSION_ENV="${RUN_DIR}/wandb-session.env"
fi
AWS_SESSION_ENV="${AWS_SESSION_ENV:-}"
if [[ -z "${AWS_SESSION_ENV}" && -n "${RUN_DIR:-}" && -f "${RUN_DIR}/aws-session.env" ]]; then
  AWS_SESSION_ENV="${RUN_DIR}/aws-session.env"
fi

if [[ -n "${WANDB_SESSION_ENV}" && -f "${WANDB_SESSION_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${WANDB_SESSION_ENV}"
fi
if [[ -n "${AWS_SESSION_ENV}" && -f "${AWS_SESSION_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${AWS_SESSION_ENV}"
fi
export AWS_SESSION_ENV

# --- task-loss dependency preflight (no network) ---
HF_SESSION_ENV="${HF_SESSION_ENV:-}"
if [[ -z "${HF_SESSION_ENV}" && -n "${RUN_DIR:-}" && -f "${RUN_DIR}/hf-session.env" ]]; then
  HF_SESSION_ENV="${RUN_DIR}/hf-session.env"
fi
if [[ -n "${HF_SESSION_ENV}" && -f "${HF_SESSION_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${HF_SESSION_ENV}"
fi
if [[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN}}"
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
  echo "[launch_arm] HF token present for ladder eval downloads" >&2
else
  echo "[launch_arm] WARNING: no HF token; public Hub downloads may be rate-limited" >&2
fi
if [[ "${TASK_LOSS_EVAL}" != "0" ]]; then
  : "${LADDER_BASE_CONFIG:?LADDER_BASE_CONFIG is required when TASK_LOSS_EVAL is enabled}"
  if [[ ! -f "${LADDER_BASE_CONFIG}" ]]; then
    echo "missing LADDER_BASE_CONFIG: ${LADDER_BASE_CONFIG}" >&2
    exit 2
  fi
  export LADDER_BASE_CONFIG
fi

if [[ "${ALLOW_LOCAL_ONLY}" != "1" ]]; then
  if [[ "${WANDB_MODE}" != "online" ]]; then
    echo "WANDB_MODE=online is required for production artifact durability" >&2
    echo "Use WANDB_MODE=disabled ALLOW_LOCAL_ONLY=1 only for local smoke." >&2
    exit 2
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY missing (required for fail-closed checkpoint uploads)" >&2
    echo "Push with: bash scripts/farmshare/push_wandb_session_to_farmshare.sh \"\${RUN_DIR}\"" >&2
    exit 2
  fi
fi

if [[ "${FRESH:-0}" == "1" && -n "${LOAD_PATH:-}" ]]; then
  echo "FRESH=1 and LOAD_PATH are mutually exclusive" >&2
  exit 2
fi
if [[ "${FRESH:-0}" != "1" && -z "${LOAD_PATH:-}" ]]; then
  echo "choose recovery mode explicitly: set FRESH=1 or LOAD_PATH=<stepN>" >&2
  exit 2
fi

export WANDB_PROJECT WANDB_ENTITY WANDB_RUN_NAME WANDB_MODE
export WANDB_DIR="${WANDB_DIR:-$(dirname "${PROGRESS_DIR}")/wandb}"
mkdir -p "${WANDB_DIR}"

ARGS=(
  --name "${ARM_ID}"
  --arm-id "${ARM_ID}"
  --pacing "${PACING}"
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --task-loss-results-dir "${TASK_LOSS_OUT_DIR}"
  --metrics-dir "${METRICS_DIR}"
  --device-batch-size "${DEVICE_BATCH_SIZE}"
  --seed "${SEED}"
  --lr-alpha-f "${LR_ALPHA_F}"
  --train-dataset-id "${TRAIN_DATASET_ID}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode "${WANDB_MODE}"
  --wandb-run-name "${WANDB_RUN_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ "${WANDB_UPLOAD_EXISTING}" == "1" ]]; then
  ARGS+=(--wandb-upload-existing)
fi
if [[ "${ALLOW_LOCAL_ONLY}" == "1" ]]; then
  ARGS+=(--allow-local-only)
fi

if [[ -n "${TRAIN_DATASET_VERSION:-}" ]]; then
  ARGS+=(--train-dataset-version "${TRAIN_DATASET_VERSION}")
fi
if [[ -n "${DATA_CACHE_DIR:-}" ]]; then
  mkdir -p "${DATA_CACHE_DIR}"
  ARGS+=(--data-cache-dir "${DATA_CACHE_DIR}")
fi
if [[ -n "${TRAIN_PATHS_FILE:-}" ]]; then
  ARGS+=(--train-paths-file "${TRAIN_PATHS_FILE}")
fi
if [[ "${FRESH:-0}" == "1" ]]; then
  ARGS+=(--fresh)
fi
if [[ "${TASK_LOSS_EVAL}" == "0" ]]; then
  ARGS+=(--no-task-loss-on-save)
else
  ARGS+=(--ladder-base-config "${LADDER_BASE_CONFIG}")
fi
if [[ -n "${LOAD_PATH:-}" ]]; then
  ARGS+=(--load-path "${LOAD_PATH}")
fi
if [[ "${PACING}" == "control" ]]; then
  :
else
  : "${DIFFICULTY_METRIC:?DIFFICULTY_METRIC is required for curriculum pacing}"
  ARGS+=(--difficulty-metric "${DIFFICULTY_METRIC}")
  if [[ -n "${CURRICULUM_INDEX:-}" ]]; then
    echo "CURRICULUM_INDEX is rejected: publish parent_pool_flat_chunks_v1 orders" >&2
    exit 2
  fi
  if [[ -n "${CURRICULUM_DATASET_ID:-}" ]]; then
    ARGS+=(--curriculum-dataset-id "${CURRICULUM_DATASET_ID}")
  fi
  if [[ -n "${CURRICULUM_DATASET_VERSION:-}" ]]; then
    ARGS+=(--curriculum-dataset-version "${CURRICULUM_DATASET_VERSION}")
  fi
  if [[ -n "${CURRICULUM_ORDER_GROUP:-}" ]]; then
    ARGS+=(--curriculum-order-group "${CURRICULUM_ORDER_GROUP}")
  fi
fi

# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS} )

echo "[launch_arm] ARM_ID=${ARM_ID} PACING=${PACING} NPROC=${NPROC} DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE} TRAIN_DATASET_ID=${TRAIN_DATASET_ID} SAVE_FOLDER=${SAVE_FOLDER}"
echo "[launch_arm] WANDB_PROJECT=${WANDB_PROJECT} WANDB_MODE=${WANDB_MODE} WANDB_RUN_NAME=${WANDB_RUN_NAME} WANDB_DIR=${WANDB_DIR}"
echo "[launch_arm] artifacts=scratch+wandb s3=published-input-staging-only"

if [[ "${NPROC}" -le 1 ]]; then
  exec "${PYTHON}" "${TRAIN_SCRIPT}" "${ARGS[@]}" "${EXTRA[@]}"
else
  exec torchrun --standalone --nproc_per_node="${NPROC}" \
    "${TRAIN_SCRIPT}" "${ARGS[@]}" "${EXTRA[@]}"
fi

