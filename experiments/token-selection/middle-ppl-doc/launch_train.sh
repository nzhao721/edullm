#!/usr/bin/env bash
# Launch Middle-PPL document CE on 1..N GPUs (ephemeral empty-scratch OK).
#
# Required env (job-scoped scratch; no host-persistent defaults):
#   STAGE_DIR         — empty-ok local/scratch staging root for edullm-data shards
#   SAVE_FOLDER       — job-scoped checkpoint scratch (uploaded to W&B after each save)
#   PROGRESS_DIR      — job-scoped metrics / run_meta (uploaded to W&B)
#
# Optional:
#   DATASET_ID        — edullm-data id (default: pretrain/middle-ppl-doc-mid60; fail-closed)
#   DATASET_VERSION   — pin version; default resolve_latest
#   DATASET_SPLIT     — default train
#   DATASET_GROUP     — payload group when multi-group
#   NPROC             — processes per node (default: 1, or WORLD_SIZE)
#   NAME              — run id (default: edullm-370M-middle-ppl-doc-ladder125-v1)
#   LENGTH_TOKENS     — default 9900000000 → 2360 steps
#   FRESH             — 1 to pass --fresh (default: 1; ephemeral starts clean)
#   LOAD_PATH         — explicit local checkpoint directory
#   WANDB_RESUME_ARTIFACT — checkpoint artifact ref for an empty scratch resume
#   ALLOW_LOCAL_ONLY  — 1 to allow debug runs without W&B durability
#   TASK_LOSS_EVAL    — 0 to disable post-save evals (default: enabled by trainer)
#   EXTRA_ARGS        — extra flags forwarded to the trainer
#
# Resolves train shards from s3://edullm-data/ via edullm_data.read; stages into STAGE_DIR.
# Durable artifacts stay on scratch and upload to W&B.
# Does not read edullm-datasets or assume pre-existing scratch corpora/checkpoints.
#
# W&B project token-selection is the artifact and metrics store. Push
# wandb-session.env via scripts/farmshare/push_wandb_session_to_farmshare.sh
# "$RUN_DIR" (or set WANDB_SESSION_ENV). Local smoke: WANDB_MODE=disabled.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${TS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${STAGE_DIR:?Set STAGE_DIR to a writable job-scoped staging directory (may start empty)}"
: "${SAVE_FOLDER:?Set SAVE_FOLDER to a job-scoped checkpoint scratch directory}"
: "${PROGRESS_DIR:?Set PROGRESS_DIR to a job-scoped progress/metrics directory}"

DATASET_ID="${DATASET_ID:-pretrain/middle-ppl-doc-mid60}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
NAME="${NAME:-edullm-370M-middle-ppl-doc-ladder125-v1}"
LENGTH_TOKENS="${LENGTH_TOKENS:-9900000000}"
FRESH="${FRESH:-1}"

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
if [[ "${TASK_LOSS_EVAL:-1}" == "0" && "${ALLOW_LOCAL_ONLY:-0}" != "1" ]]; then
  echo "TASK_LOSS_EVAL=0 is permitted only with ALLOW_LOCAL_ONLY=1 smoke runs." >&2
  exit 1
fi

if [[ -n "${LOAD_PATH:-}" && "${FRESH}" == "1" ]]; then
  echo "Refusing FRESH=1 together with LOAD_PATH=${LOAD_PATH}" >&2
  exit 1
fi

TRAINER="${SCRIPT_DIR}/train_ce_middle_ppl_doc.py"
COMMON_ARGS=(
  --name "${NAME}"
  --dataset-id "${DATASET_ID}"
  --dataset-split "${DATASET_SPLIT}"
  --stage-dir "${STAGE_DIR}"
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --length-tokens "${LENGTH_TOKENS}"
)

if [[ -n "${DATASET_VERSION:-}" ]]; then
  COMMON_ARGS+=(--dataset-version "${DATASET_VERSION}")
fi
if [[ -n "${DATASET_GROUP:-}" ]]; then
  COMMON_ARGS+=(--dataset-group "${DATASET_GROUP}")
fi

# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS:-} )

if [[ -n "${TASK_LOSS_OUT_DIR:-}" ]]; then
  EXTRA+=(--task-loss-results-dir "${TASK_LOSS_OUT_DIR}")
fi
if [[ -n "${TASK_LOSS_EVAL_SCRIPT:-}" ]]; then
  EXTRA+=(--task-loss-eval-script "${TASK_LOSS_EVAL_SCRIPT}")
fi
if [[ "${TASK_LOSS_EVAL:-1}" == "0" ]]; then
  EXTRA+=(--no-task-loss-on-save)
fi
if [[ "${FRESH}" == "1" ]]; then
  EXTRA+=(--fresh)
fi
if [[ -n "${LOAD_PATH:-}" ]]; then
  EXTRA+=(--load-path "${LOAD_PATH}")
fi
if [[ -n "${WANDB_RESUME_ARTIFACT:-}" ]]; then
  EXTRA+=(--wandb-resume-artifact "${WANDB_RESUME_ARTIFACT}")
fi
if [[ "${ALLOW_LOCAL_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--allow-local-only)
fi

echo "Launching middle-ppl-doc: nproc_per_node=${NPROC_PER_NODE} name=${NAME} dataset=${DATASET_ID}" >&2

# shellcheck disable=SC1091
source "${TS_ROOT}/token_selection/scripts/wandb_env.sh" "middle-ppl-doc" "${NAME}"

if [[ "${NPROC_PER_NODE}" -eq 1 ]] && [[ -z "${WORLD_SIZE:-}" ]]; then
  exec python "${TRAINER}" "${COMMON_ARGS[@]}" "${EXTRA[@]}"
fi

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${TRAINER}" \
  "${COMMON_ARGS[@]}" \
  "${EXTRA[@]}"
