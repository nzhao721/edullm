#!/usr/bin/env bash
# Launch RefHQ OLMo-2 370M reference training (train_olmo3_370m_refhq.py only).
#
# Clean / ephemeral machine: scratch starts empty. Stage from s3://edullm-data,
# train with local SAVE_FOLDER on scratch, upload durable checkpoints to
# s3://edullm-checkpoints/token-sel/reference/ (disable with S3_EXPORT=0).
#
# W&B (SmolLM2 protocol; additive to S3): project token-selection. Push
# wandb-session.env via scripts/farmshare/push_wandb_session_to_farmshare.sh
# "$RUN_DIR" (or set WANDB_SESSION_ENV). Local smoke: WANDB_MODE=disabled.
#
# Does not assume FarmShare scratch, laptop trees, or legacy s3://edullm-datasets/.
#
# Required:
#   SAVE_FOLDER     — ephemeral checkpoint working dir (uploaded to S3)
#   PROGRESS_DIR    — ephemeral metrics / run_meta (uploaded to S3)
#
# Data (one of):
#   STAGE_DIR       — scratch dir; trainer downloads edullm-data shards here
#   REFHQ_WORK      — if set, run prepare_refhq_data.py first, then pass paths
#   PATHS_FILE      — reuse paths_train.txt from a prior stage of this arm
#   STAGE_DIR=      — empty string: train from s3:// URIs (no local stage)
#
# Optional:
#   NPROC           — processes per node (default: 1, or WORLD_SIZE if set)
#   NAME            — run id (default: refhq-regmix-5p5b-v1)
#   DATASET_ID      — default pretrain/refhq-regmix-5p5b
#   DATASET_VERSION — pin version (default: resolve_latest)
#   S3_EXPORT       — 0 to disable live checkpoint uploads (default: on)
#   EXTRA_ARGS      — extra flags forwarded to the trainer
#
# Examples:
#   # Ephemeral: stage + train + S3 export
#   STAGE_DIR=$SCRATCH/staged SAVE_FOLDER=$SCRATCH/ckpts PROGRESS_DIR=$SCRATCH/progress \
#     NPROC=4 ./launch_train.sh
#
#   # Explicit prepare then train
#   REFHQ_WORK=$SCRATCH/refhq SAVE_FOLDER=$SCRATCH/ckpts PROGRESS_DIR=$SCRATCH/progress \
#     ./launch_train.sh
#
#   # Direct s3:// URIs (no local stage):
#   SAVE_FOLDER=$SCRATCH/ckpts PROGRESS_DIR=$SCRATCH/progress STAGE_DIR= ./launch_train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${TS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${SAVE_FOLDER:?Set SAVE_FOLDER to an ephemeral checkpoint working directory}"
: "${PROGRESS_DIR:?Set PROGRESS_DIR to an ephemeral progress/metrics directory}"

NAME="${NAME:-refhq-regmix-5p5b-v1}"
DATASET_ID="${DATASET_ID:-pretrain/refhq-regmix-5p5b}"

# Optional prepare into REFHQ_WORK (writes tokenized/paths_train.txt).
if [[ -n "${REFHQ_WORK:-}" ]]; then
  echo "launch_train: staging RefHQ into REFHQ_WORK=${REFHQ_WORK} from edullm-data" >&2
  PREPARE_ARGS=(--work "${REFHQ_WORK}" --dataset-id "${DATASET_ID}")
  if [[ -n "${DATASET_VERSION:-}" ]]; then
    PREPARE_ARGS+=(--dataset-version "${DATASET_VERSION}")
  fi
  python3 "${SCRIPT_DIR}/prepare_refhq_data.py" "${PREPARE_ARGS[@]}"
  PATHS_FILE="${PATHS_FILE:-${REFHQ_WORK}/tokenized/paths_train.txt}"
fi

# If STAGE_DIR unset and no paths file: default under PROGRESS_DIR (still ephemeral).
if [[ -z "${PATHS_FILE:-}" ]] && [[ -z "${STAGE_DIR+x}" ]]; then
  STAGE_DIR="${PROGRESS_DIR}/staged-refhq"
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

TRAINER="${SCRIPT_DIR}/train_olmo3_370m_refhq.py"
COMMON_ARGS=(
  --name "${NAME}"
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --dataset-id "${DATASET_ID}"
)

if [[ -n "${DATASET_VERSION:-}" ]]; then
  COMMON_ARGS+=(--dataset-version "${DATASET_VERSION}")
fi

if [[ -n "${PATHS_FILE:-}" ]]; then
  COMMON_ARGS+=(--paths-file "${PATHS_FILE}")
elif [[ -n "${STAGE_DIR:-}" ]]; then
  mkdir -p "${STAGE_DIR}"
  COMMON_ARGS+=(--stage-dir "${STAGE_DIR}")
fi

# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS:-} )

echo "Launching RefHQ reference: nproc_per_node=${NPROC_PER_NODE} name=${NAME} dataset=${DATASET_ID}${DATASET_VERSION:+/${DATASET_VERSION}} stage=${STAGE_DIR:-<s3-uris|paths-file>} s3_export=${S3_EXPORT:-1}" >&2

# shellcheck disable=SC1091
source "${TS_ROOT}/token_selection/scripts/wandb_env.sh" "reference" "${NAME}"

set +e
if [[ "${NPROC_PER_NODE}" -eq 1 ]] && [[ -z "${WORLD_SIZE:-}" ]]; then
  python "${TRAINER}" "${COMMON_ARGS[@]}" "${EXTRA[@]}"
  rc=$?
else
  torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${TRAINER}" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA[@]}"
  rc=$?
fi
set -e

# Upload-before-end belt: fail-closed when S3_EXPORT is on (local smoke: S3_EXPORT=0).
if [[ "${rc}" -eq 0 ]] && [[ "${S3_EXPORT:-1}" != "0" ]]; then
  if ! command -v aws >/dev/null 2>&1; then
    echo "launch_train: durable S3 export required but aws CLI missing; set S3_EXPORT=0 for local smoke only" >&2
    exit 1
  fi
  REMOTE_CKPT="s3://edullm-checkpoints/token-sel/reference/checkpoints/${NAME}/"
  REMOTE_PROG="s3://edullm-checkpoints/token-sel/reference/progress/${NAME}/"
  echo "launch_train: final S3 sync → ${REMOTE_CKPT}" >&2
  aws s3 sync "${SAVE_FOLDER}" "${REMOTE_CKPT}" --only-show-errors
  aws s3 sync "${PROGRESS_DIR}" "${REMOTE_PROG}" --only-show-errors
fi

exit "${rc}"
