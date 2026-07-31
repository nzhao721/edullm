#!/usr/bin/env bash
# Launch one curriculum / control arm on an ephemeral runtime.
#
# Scratch is assumed empty at job start and wiped after. Stage train/curriculum
# bytes from s3://edullm-data/ into a job-scoped cache. Durable artifacts go to
# s3://edullm-checkpoints/curriculum/<arm_id>/ (trainer --s3-export, default on;
# export failure aborts the job — opt out only with S3_EXPORT=0 for local smoke)
# and/or W&B project "curriculum" (SmolLM FarmShare protocol).
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
#   CURRICULUM_DATASET_ID      curriculum arms: default mapped from DIFFICULTY_METRIC
#   CURRICULUM_DATASET_VERSION optional pin
#   DATA_CACHE_DIR / EDULLM_DATA_CACHE  job-scoped staging root for fetch-if-missing
#   TRAIN_PATHS_FILE           optional: paths already staged in THIS job
#   CURRICULUM_INDEX           optional: local index staged in THIS job
#
# W&B (SmolLM protocol; durable sink with S3):
#   RUN_DIR             job scratch root; sources ${RUN_DIR}/wandb-session.env + aws-session.env
#   WANDB_PROJECT       default: curriculum
#   WANDB_ENTITY        optional (empty = account default; do not invent)
#   WANDB_RUN_NAME      default: ARM_ID (or ${ARM_ID}-${SLURM_JOB_ID} when set)
#   WANDB_MODE          default: online (requires wandb-session.env); local smoke: disabled
#   WANDB_UPLOAD_EXISTING=1  upload existing local ckpts/evals on start
#   ALLOW_LOCAL_ONLY=1  scratch-only escape hatch (with WANDB_MODE=disabled + S3_EXPORT=0)
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
#   FRESH=1             ignore leftover local ckpts under SAVE_FOLDER
#   LOAD_PATH           local step dir, or s3://edullm-checkpoints/curriculum/<arm>/checkpoints/stepN
#   TASK_LOSS_OUT_DIR   default: $PROGRESS_DIR/task_loss_results
#   METRICS_DIR         default: sibling of PROGRESS_DIR named metrics
#   S3_EXPORT=0 / SKIP_S3_UPLOAD=1  local-smoke only (disables fail-closed durable upload)
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
#     SAVE_FOLDER=$RUN_DIR/ckpts PROGRESS_DIR=$RUN_DIR/progress \
#     DATA_CACHE_DIR=$RUN_DIR/cache \
#     bash launch_arm.sh
#
# Example (N ranks, curriculum — requires published curriculum/* on edullm-data):
#   NPROC=4 ARM_ID=linear10-cr PACING=linear_n10 DIFFICULTY_METRIC=compression_ratio \
#     SAVE_FOLDER=$RUN_DIR/ckpts PROGRESS_DIR=$RUN_DIR/progress \
#     DATA_CACHE_DIR=$RUN_DIR/cache \
#     bash launch_arm.sh
#
# Local smoke (no durable sink):
#   WANDB_MODE=disabled S3_EXPORT=0 ALLOW_LOCAL_ONLY=1 bash launch_arm.sh
#
# Resume (pull from S3 into SAVE_FOLDER, or pass a local staged dir):
#   LOAD_PATH=s3://edullm-checkpoints/curriculum/$ARM_ID/checkpoints/step1250 \
#     bash launch_arm.sh
#
# Optional post-hoc EMA (after pulling checkpoints from S3 into a work dir):
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

if [[ "${WANDB_MODE}" == "online" && "${ALLOW_LOCAL_ONLY}" != "1" ]]; then
  if [[ -z "${WANDB_SESSION_ENV}" || ! -f "${WANDB_SESSION_ENV}" ]]; then
    echo "missing wandb-session.env (required for WANDB_MODE=online durable upload)" >&2
    echo "Push with: bash scripts/farmshare/push_wandb_session_to_farmshare.sh \"\${RUN_DIR}\"" >&2
    echo "Or set WANDB_MODE=disabled S3_EXPORT=0 ALLOW_LOCAL_ONLY=1 for local smoke." >&2
    exit 2
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY missing after sourcing wandb-session.env" >&2
    exit 2
  fi
fi

if [[ "${S3_EXPORT:-1}" == "0" || "${SKIP_S3_UPLOAD:-0}" == "1" ]]; then
  if [[ "${WANDB_MODE}" != "online" && "${ALLOW_LOCAL_ONLY}" != "1" ]]; then
    echo "durable save required: keep S3_EXPORT=1 and/or WANDB_MODE=online, or set ALLOW_LOCAL_ONLY=1" >&2
    exit 2
  fi
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
if [[ -n "${LOAD_PATH:-}" ]]; then
  ARGS+=(--load-path "${LOAD_PATH}")
fi
if [[ "${S3_EXPORT:-1}" == "0" || "${SKIP_S3_UPLOAD:-0}" == "1" ]]; then
  ARGS+=(--no-s3-export)
fi

if [[ "${PACING}" == "control" ]]; then
  :
else
  : "${DIFFICULTY_METRIC:?DIFFICULTY_METRIC is required for curriculum pacing}"
  ARGS+=(--difficulty-metric "${DIFFICULTY_METRIC}")
  if [[ -n "${CURRICULUM_INDEX:-}" ]]; then
    ARGS+=(--curriculum-index "${CURRICULUM_INDEX}")
  fi
  if [[ -n "${CURRICULUM_DATASET_ID:-}" ]]; then
    ARGS+=(--curriculum-dataset-id "${CURRICULUM_DATASET_ID}")
  fi
  if [[ -n "${CURRICULUM_DATASET_VERSION:-}" ]]; then
    ARGS+=(--curriculum-dataset-version "${CURRICULUM_DATASET_VERSION}")
  fi
fi

# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS} )

echo "[launch_arm] ARM_ID=${ARM_ID} PACING=${PACING} NPROC=${NPROC} DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE} TRAIN_DATASET_ID=${TRAIN_DATASET_ID} SAVE_FOLDER=${SAVE_FOLDER}"
echo "[launch_arm] WANDB_PROJECT=${WANDB_PROJECT} WANDB_MODE=${WANDB_MODE} WANDB_RUN_NAME=${WANDB_RUN_NAME} WANDB_DIR=${WANDB_DIR}"

if [[ "${NPROC}" -le 1 ]]; then
  exec "${PYTHON}" "${TRAIN_SCRIPT}" "${ARGS[@]}" "${EXTRA[@]}"
else
  exec torchrun --standalone --nproc_per_node="${NPROC}" \
    "${TRAIN_SCRIPT}" "${ARGS[@]}" "${EXTRA[@]}"
fi

