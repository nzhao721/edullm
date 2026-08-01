#!/usr/bin/env bash
# Launch learnability-doc CE train (1..N GPUs via torchrun / WORLD_SIZE).
# Does not hardcode GPU count, device IDs, or host paths.
#
# Ephemeral empty-scratch contract:
#   - Stage train shards from s3://edullm-data/ into STAGE_DIR for this job.
#   - Do not assume pre-existing scratch corpora, venvs, or local checkpoints.
#   - Artifacts remain on scratch and upload to W&B (production online fail-closed).
#   - Resume via WANDB_RESUME_ARTIFACT or an explicit local LOAD_PATH.
#
# W&B project token-selection is the artifact and metrics store. Push
# wandb-session.env via scripts/farmshare/push_wandb_session_to_farmshare.sh
# "$RUN_DIR" (or set WANDB_SESSION_ENV). Local smoke: WANDB_MODE=disabled.
#
# Required:
#   STAGE_DIR         scratch dir for clean-machine fetch from s3://edullm-data/
#                     (unless TRAIN_PATHS_FILE reuses a same-job stage)
#   SAVE_FOLDER       job-local checkpoint root (not assumed durable across wipe)
#   PROGRESS_DIR      job-local run meta / progress
#
# Optional:
#   DATASET_ID           default pretrain/learnability-doc-top60 (fail-closed if unpublished)
#   DATASET_VERSION      pin version; default resolve_latest
#   TRAIN_PATHS_FILE     reuse paths_train.txt from a prior *same-job* --stage-dir fetch
#   NPROC / WORLD_SIZE   default 1
#   NAME                 default edullm-370M-learnability-doc-10b
#   LENGTH_TOKENS        default 9900000000 (2360 steps)
#   TASK_LOSS_EVAL       0 to disable post-save evals
#   TASK_LOSS_EVAL_SCRIPT / TASK_LOSS_OUT_DIR
#   LOAD_PATH            explicit checkpoint dir to resume (no local auto-resume)
#   FRESH=1              explicit start-from-0 (default when LOAD_PATH unset)
#   WANDB_RESUME_ARTIFACT W&B checkpoint artifact ref for cross-job resume
#   EXTRA_ARGS
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${SAVE_FOLDER:?Set SAVE_FOLDER for job-local checkpoints (uploaded to W&B)}"
: "${PROGRESS_DIR:?Set PROGRESS_DIR for run meta / progress}"

if [[ -z "${TRAIN_PATHS_FILE:-}" ]]; then
  : "${STAGE_DIR:?Set STAGE_DIR (empty-scratch cache) for edullm-data fetch, or TRAIN_PATHS_FILE from a same-job stage}"
fi

NAME="${NAME:-edullm-370M-learnability-doc-10b}"
LENGTH_TOKENS="${LENGTH_TOKENS:-9900000000}"
DATASET_ID="${DATASET_ID:-pretrain/learnability-doc-top60}"
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

export TASK_LOSS_STRICT=1
export TASK_LOSS_NPROC="${TASK_LOSS_NPROC:-${NPROC_PER_NODE}}"

export PYTHONPATH="${TS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

EXTRA=()
if [[ -n "${LOAD_PATH:-}" ]]; then
  EXTRA+=(--load-path "${LOAD_PATH}")
elif [[ "${FRESH:-1}" == "1" ]]; then
  EXTRA+=(--fresh)
fi
if [[ "${NO_TASK_LOSS:-0}" == "1" ]] || [[ "${TASK_LOSS_EVAL:-1}" == "0" ]]; then
  EXTRA+=(--no-task-loss-on-save)
fi
if [[ -n "${TASK_LOSS_EVAL_SCRIPT:-}" ]]; then
  EXTRA+=(--task-loss-eval-script "${TASK_LOSS_EVAL_SCRIPT}")
fi
if [[ -n "${DATASET_VERSION:-}" ]]; then
  EXTRA+=(--dataset-version "${DATASET_VERSION}")
fi
if [[ -n "${STAGE_DIR:-}" ]]; then
  EXTRA+=(--stage-dir "${STAGE_DIR}")
fi
if [[ -n "${TRAIN_PATHS_FILE:-}" ]]; then
  EXTRA+=(--train-paths-file "${TRAIN_PATHS_FILE}")
fi
if [[ -n "${WANDB_RESUME_ARTIFACT:-}" ]]; then
  EXTRA+=(--wandb-resume-artifact "${WANDB_RESUME_ARTIFACT}")
fi
# shellcheck disable=SC2206
EXTRA+=( ${EXTRA_ARGS:-} )

COMMON_ARGS=(
  --name "${NAME}"
  --dataset-id "${DATASET_ID}"
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --length-tokens "${LENGTH_TOKENS}"
  --task-loss-results-dir "${TASK_LOSS_RESULTS_DIR}"
)

TRAINER="${SCRIPT_DIR}/train_ce_learnability_doc_olmo_370m.py"
echo "Launching learnability-doc: nproc_per_node=${NPROC_PER_NODE} name=${NAME} dataset=${DATASET_ID}" >&2

# shellcheck disable=SC1091
source "${TS_ROOT}/token_selection/scripts/wandb_env.sh" "learnability-doc" "${NAME}"

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
