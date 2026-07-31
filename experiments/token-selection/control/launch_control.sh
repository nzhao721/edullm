#!/usr/bin/env bash
# Launch Control (random 60% keep on RegMix 10B) on 1..N GPUs.
#
# Ephemeral runtime: scratch may start empty and be wiped after the job.
# Data comes from published+validated s3://edullm-data/pretrain/regmix-10b
# (edullm_data.read.resolve_latest + dataset_paths). Rank 0 stages shards into
# STAGE_DIR. Optional TRAIN_PATHS_FILE reuses a prior stage of this arm.
# Never reads s3://edullm-datasets/ or assumes FarmShare/laptop corpora exist.
#
# Durable artifacts: trainer syncs checkpoints/progress/task_loss to
# s3://edullm-checkpoints/token-sel/control/ (upload-before-end; fail-closed
# when S3 export is enabled). Disable with S3_EXPORT=0 / SKIP_S3_UPLOAD=1.
#
# W&B (SmolLM2 protocol; additive to S3): project token-selection. Push
# wandb-session.env via scripts/farmshare/push_wandb_session_to_farmshare.sh
# "$RUN_DIR" (or set WANDB_SESSION_ENV). Local smoke: WANDB_MODE=disabled.
#
# Required env / args (no host-specific defaults):
#   SAVE_FOLDER       — local/scratch checkpoint root (synced to S3)
#   PROGRESS_DIR      — local/scratch metrics / run_meta (synced to S3)
#   STAGE_DIR         — ephemeral scratch for edullm-data shards
#                       (unless TRAIN_PATHS_FILE from a prior stage)
#
# Optional:
#   TRAIN_PATHS_FILE  — prior-stage memmap path list (skips S3 stage)
#   DATASET_ID        — default: pretrain/regmix-10b
#   DATASET_VERSION   — pin version (default: resolve_latest)
#   NPROC             — processes per node (default: 1, or WORLD_SIZE if set)
#   NAME              — run id (default: control-regmix10b-v2)
#   LENGTH_TOKENS     — default 9900000000 → 2360 steps (one-epoch matrix)
#   FRESH=1           — ignore any local checkpoints
#   LOAD_PATH         — explicit resume dir (stage from S3 first on clean machines)
#   ALLOW_LOCAL_RESUME=1 — same-job local auto-resume only (not across wiped scratch)
#   TASK_LOSS_EVAL    — 0 to disable post-save evals (default: enabled by trainer)
#   S3_EXPORT         — 0 to disable durable S3 sync (default: on when aws CLI present)
#   EXTRA_ARGS        — extra flags forwarded to the trainer
#
# Examples:
#   # Clean ephemeral machine: resolve+stage from edullm-data, then train
#   STAGE_DIR=.../regmix-10b SAVE_FOLDER=.../ckpts PROGRESS_DIR=.../progress \
#     NPROC=1 ./launch_control.sh
#
#   # Optional pre-stage, then train
#   python prepare_control_data.py --work "$STAGE_DIR"
#   TRAIN_PATHS_FILE=$STAGE_DIR/train_tokenized/paths_train.txt \
#     SAVE_FOLDER=... PROGRESS_DIR=... NPROC=4 ./launch_control.sh
#
#   # Resume: stage ckpt from s3://edullm-checkpoints/token-sel/control/ first
#   LOAD_PATH=.../step1250 SAVE_FOLDER=... PROGRESS_DIR=... STAGE_DIR=... \
#     ./launch_control.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TS_ROOT}/../.." && pwd)"
export PYTHONPATH="${TS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${SAVE_FOLDER:?Set SAVE_FOLDER to the (ephemeral) checkpoint directory}"
: "${PROGRESS_DIR:?Set PROGRESS_DIR to the (ephemeral) progress/metrics directory}"

NAME="${NAME:-control-regmix10b-v2}"
LENGTH_TOKENS="${LENGTH_TOKENS:-9900000000}"
DATASET_ID="${DATASET_ID:-pretrain/regmix-10b}"

if [[ -z "${TRAIN_PATHS_FILE:-}" ]]; then
  : "${STAGE_DIR:?Set STAGE_DIR (ephemeral scratch) for edullm-data fetch, or TRAIN_PATHS_FILE from a prior stage}"
fi

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
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --length-tokens "${LENGTH_TOKENS}"
  --dataset-id "${DATASET_ID}"
)

if [[ -n "${TRAIN_PATHS_FILE:-}" ]]; then
  COMMON_ARGS+=(--train-paths-file "${TRAIN_PATHS_FILE}")
fi
if [[ -n "${STAGE_DIR:-}" ]]; then
  COMMON_ARGS+=(--stage-dir "${STAGE_DIR}")
fi
if [[ -n "${DATASET_VERSION:-}" ]]; then
  COMMON_ARGS+=(--dataset-version "${DATASET_VERSION}")
fi
if [[ "${FRESH:-0}" == "1" ]]; then
  COMMON_ARGS+=(--fresh)
fi
if [[ -n "${LOAD_PATH:-}" ]]; then
  COMMON_ARGS+=(--load-path "${LOAD_PATH}")
fi
if [[ "${TASK_LOSS_EVAL:-1}" == "0" ]] || [[ "${NO_TASK_LOSS:-0}" == "1" ]]; then
  COMMON_ARGS+=(--no-task-loss-on-save)
fi

# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS:-} )

echo "Launching control: nproc_per_node=${NPROC_PER_NODE} name=${NAME} dataset=${DATASET_ID}" >&2

# shellcheck disable=SC1091
source "${TS_ROOT}/token_selection/scripts/wandb_env.sh" "control" "${NAME}"

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
