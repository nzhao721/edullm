#!/usr/bin/env bash
# Launch one OLMo2-370M CE arm on a mixlaw 370M validation mixture.
#
# Ephemeral / empty-scratch friendly:
#   Domain weights from validation_mixtures_10b.json (via mix_weights.json).
#   Streams from a working pool staged from published s3://edullm-data/
#   (default: pretrain/olmo-127b). Never reads s3://edullm-datasets/.
#   Durable artifacts: trainer fail-closed syncs to
#     s3://edullm-checkpoints/mixlaw/370m-validation/<MIX_NAME>/
#   (missing aws/creds or sync failure aborts). Opt out: S3_EXPORT=0 / --no-s3-export.
#   Resume: EXTRA_ARGS='--load-path s3://edullm-checkpoints/mixlaw/370m-validation/<mix>/checkpoints/stepN'
#
# Required:
#   MIX_NAME            e.g. ML-pilot_caps | mix01 | mix07
#   MIX_WEIGHTS_JSON    per-arm sidecar from prepare_validation_370m_data.py
#   SAVE_FOLDER         job-scoped checkpoint root (wiped with scratch)
#   PROGRESS_DIR        job-scoped metrics / run_meta
#
# Optional:
#   DATASET_ID          edullm-data id (default pretrain/olmo-127b)
#   DATASET_VERSION     pin version; default resolve_latest
#   POOL_DIR            pre-staged working pool; if unset/incomplete, auto-stage
#   STAGE_DIR           staging target when POOL_DIR is absent
#   NPROC               ranks (default 1; >1 → torchrun)
#   LENGTH_TOKENS       default 10000000000
#   S3_EXPORT           0 to disable live aws s3 sync (default: enabled)
#   EXTRA_ARGS          forwarded to train_mixlaw_validation_370m.py
#   PYTHON              python executable (default: python)
#   WANDB_MODE          online|offline|disabled (default: online if
#                       wandb-session.env / WANDB_API_KEY present else disabled)
#   WANDB_PROJECT       default mixlaw
#
# Mint W&B session (optional, additive to S3):
#   bash scripts/farmshare/push_wandb_session_to_farmshare.sh "$RUN_DIR"
#
# Example (clean machine — let the trainer stage from edullm-data):
#   MIX_NAME=ML-near-opt-4 \
#   MIX_WEIGHTS_JSON=/work/ML-near-opt-4/mix_weights.json \
#   SAVE_FOLDER=/scratch/job/ckpts PROGRESS_DIR=/scratch/job/progress \
#   STAGE_DIR=/scratch/job/pool NPROC=4 bash launch_validation_370m.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer an explicit checkout (FarmShare ephemeral RUN_DIR copies launch into scratch).
if [[ -n "${EDULLM_ROOT:-}" && -d "${EDULLM_ROOT}/experiments/curriculum" ]]; then
  REPO_ROOT="$(cd "${EDULLM_ROOT}" && pwd)"
elif [[ -d "${SCRIPT_DIR}/../../../experiments/curriculum" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
else
  echo "[launch_validation_370m] error: set EDULLM_ROOT to the edullm checkout" >&2
  exit 2
fi
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}:${REPO_ROOT}/experiments/token-selection:${REPO_ROOT}/experiments/curriculum"

: "${MIX_NAME:?Set MIX_NAME to a validation_mixtures_10b.json run_name}"
: "${MIX_WEIGHTS_JSON:?Set MIX_WEIGHTS_JSON to the arm mix_weights.json}"
: "${SAVE_FOLDER:?Set SAVE_FOLDER}"
: "${PROGRESS_DIR:?Set PROGRESS_DIR}"

DATASET_ID="${DATASET_ID:-pretrain/olmo-127b}"
NPROC="${NPROC:-1}"
LENGTH_TOKENS="${LENGTH_TOKENS:-10000000000}"
NAME="${NAME:-mixlaw-370m-${MIX_NAME}}"
PYTHON="${PYTHON:-python}"
S3_EXPORT="${S3_EXPORT:-1}"
export S3_EXPORT

WANDB_PROJECT="${WANDB_PROJECT:-mixlaw}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-370m-validation}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${NAME}}"
# Session may live next to SAVE/PROGRESS parent (RUN_DIR) or CWD.
_WANDB_SESSION=""
for _cand in \
  "${RUN_DIR:-}/wandb-session.env" \
  "$(dirname "${PROGRESS_DIR}")/wandb-session.env" \
  "$(dirname "${SAVE_FOLDER}")/wandb-session.env" \
  "./wandb-session.env"
do
  if [[ -n "${_cand}" && -f "${_cand}" ]]; then
    _WANDB_SESSION="${_cand}"
    break
  fi
done
if [[ -n "${_WANDB_SESSION}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${_WANDB_SESSION}"
  set -u
fi
if [[ -f "${_WANDB_SESSION:-}" || -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_MODE="${WANDB_MODE:-online}"
else
  WANDB_MODE="${WANDB_MODE:-disabled}"
fi
if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "[launch_validation_370m] WANDB_MODE=online but no API key; using disabled" >&2
  WANDB_MODE=disabled
fi
export WANDB_DIR="${PROGRESS_DIR}/wandb"
export WANDB_PROJECT WANDB_MODE
mkdir -p "${WANDB_DIR}"

TRAINER="${SCRIPT_DIR}/train_mixlaw_validation_370m.py"
COMMON_ARGS=(
  --name "${NAME}"
  --mix-name "${MIX_NAME}"
  --dataset-id "${DATASET_ID}"
  --mix-weights-json "${MIX_WEIGHTS_JSON}"
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --length-tokens "${LENGTH_TOKENS}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode "${WANDB_MODE}"
  --wandb-run-name "${WANDB_RUN_NAME}"
  --wandb-group "${WANDB_GROUP}"
)
if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ "${S3_EXPORT}" == "0" || "${S3_EXPORT}" == "false" || "${S3_EXPORT}" == "no" || "${S3_EXPORT}" == "off" ]]; then
  COMMON_ARGS+=(--no-s3-export)
else
  COMMON_ARGS+=(--s3-export)
fi
if [[ -n "${DATASET_VERSION:-}" ]]; then
  COMMON_ARGS+=(--dataset-version "${DATASET_VERSION}")
fi
if [[ -n "${POOL_DIR:-}" ]]; then
  COMMON_ARGS+=(--pool-dir "${POOL_DIR}")
fi
if [[ -n "${STAGE_DIR:-}" ]]; then
  COMMON_ARGS+=(--stage-dir "${STAGE_DIR}")
fi
# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS:-} )

echo "[launch_validation_370m] MIX_NAME=${MIX_NAME} DATASET_ID=${DATASET_ID} POOL_DIR=${POOL_DIR:-<auto>} NPROC=${NPROC} S3_EXPORT=${S3_EXPORT} WANDB_MODE=${WANDB_MODE}" >&2

if [[ "${NPROC}" -eq 1 ]]; then
  exec "${PYTHON}" "${TRAINER}" "${COMMON_ARGS[@]}" "${EXTRA[@]}"
fi

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  "${TRAINER}" \
  "${COMMON_ARGS[@]}" \
  "${EXTRA[@]}"
