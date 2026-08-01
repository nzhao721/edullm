#!/usr/bin/env bash
# Launch one OLMo2-370M CE arm on a mixlaw 370M validation mixture.
#
# Ephemeral / empty-scratch friendly:
#   Domain weights from validation_mixtures_10b.json (via mix_weights.json).
#   Streams from a working pool staged from published s3://edullm-data/
#   (default: pretrain/olmo-127b). Never reads s3://edullm-datasets/.
#   Durable artifacts: local scratch checkpoints + W&B artifacts (fail-closed).
#   S3 is only for staging edullm-data / bootstrap inputs — not checkpoint storage.
#   Recovery is explicit: RECOVERY_MODE=fresh|resume|fail (default: fail).
#   resume requires EXTRA_ARGS='--load-path .../stepN' or durable metadata.
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
#   RECOVERY_MODE       fresh|resume|fail (default fail; never auto-selects)
#   DURABLE_METADATA_PATH local last_durable_step.json used only by resume
#   POOL_DIR            pre-staged working pool; if unset/incomplete, auto-stage
#   STAGE_DIR           staging target when POOL_DIR is absent
#   NPROC               ranks (default 1; >1 → torchrun)
#   LENGTH_TOKENS       default 10000000000
#   S3_EXPORT           legacy no-op for checkpoints (default: 0; use W&B)
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
S3_EXPORT="${S3_EXPORT:-0}"
RECOVERY_MODE="${RECOVERY_MODE:-fail}"
DURABLE_METADATA_PATH="${DURABLE_METADATA_PATH:-${PROGRESS_DIR}/last_durable_step.json}"
export S3_EXPORT

WANDB_PROJECT="${WANDB_PROJECT:-mixlaw}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-370m-validation}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${NAME}}"
# Session may live next to SAVE/PROGRESS parent (RUN_DIR) or CWD.
_WANDB_SESSION=""
if [[ "${MIXLAW_PLATFORM:-0}" != "1" ]]; then
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
fi
# HF auth for ladder eval dataset/tokenizer downloads (torchrun workers inherit).
_HF_SESSION=""
if [[ "${MIXLAW_PLATFORM:-0}" != "1" ]]; then
  for _cand in \
    "${RUN_DIR:-}/hf-session.env" \
    /workspace/hf-session.env \
    "$(dirname "${PROGRESS_DIR}")/hf-session.env" \
    "$(dirname "${SAVE_FOLDER}")/hf-session.env" \
    "./hf-session.env"
  do
    if [[ -n "${_cand}" && -f "${_cand}" ]]; then
      _HF_SESSION="${_cand}"
      break
    fi
  done
  if [[ -n "${_HF_SESSION}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${_HF_SESSION}"
    set -u
  fi
fi
if [[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN}}"
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
  if [[ "${MIXLAW_PLATFORM:-0}" != "1" ]]; then
    mkdir -p "${HOME}/.cache/huggingface"
    printf '%s' "${HF_TOKEN}" > "${HOME}/.cache/huggingface/token"
  fi
  echo "[launch_validation_370m] HF token present (auth for eval downloads)" >&2
else
  echo "[launch_validation_370m] WARNING: no HF_TOKEN; eval HF downloads may be rate-limited" >&2
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
PREFLIGHT="${SCRIPT_DIR}/preflight_validation_370m.py"
RECOVERY_HELPER="${SCRIPT_DIR}/mixlaw_runtime.py"
DEPENDENCY_METADATA="${PROGRESS_DIR}/dependency_versions.json"

LADDER_BASE_CONFIG="${LADDER_BASE_CONFIG:-${SCRIPT_DIR}/ladder_base_config.yaml}"
export LADDER_BASE_CONFIG
if [[ ! -f "${LADDER_BASE_CONFIG}" ]]; then
  echo "[launch_validation_370m] LADDER_BASE_CONFIG must name an existing config" >&2
  exit 2
fi
"${PYTHON}" "${PREFLIGHT}" \
  --ladder-base-config "${LADDER_BASE_CONFIG}" \
  --out "${DEPENDENCY_METADATA}"

COMMON_ARGS=(
  --name "${NAME}"
  --mix-name "${MIX_NAME}"
  --dataset-id "${DATASET_ID}"
  --mix-weights-json "${MIX_WEIGHTS_JSON}"
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --length-tokens "${LENGTH_TOKENS}"
  --recovery-mode "${RECOVERY_MODE}"
  --dependency-metadata "${DEPENDENCY_METADATA}"
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
if [[ -n "${TASK_LOSS_RESULTS_DIR:-}" ]]; then
  COMMON_ARGS+=(--task-loss-results-dir "${TASK_LOSS_RESULTS_DIR}")
fi
if [[ -n "${CHECKPOINT_PREFIX:-}" ]]; then
  COMMON_ARGS+=(--checkpoint-prefix "${CHECKPOINT_PREFIX}")
fi
if [[ -n "${OUTPUT_PREFIX:-}" ]]; then
  COMMON_ARGS+=(--output-prefix "${OUTPUT_PREFIX}")
fi
if [[ "${MIXLAW_PLATFORM:-0}" == "1" ]]; then
  COMMON_ARGS+=(--selected-arm-stage --delete-stage-shards)
fi
_recovery_cmd=(
  "${PYTHON}" "${RECOVERY_HELPER}"
  --mode "${RECOVERY_MODE}"
  --mix-name "${MIX_NAME}"
  --extra-args "${EXTRA_ARGS:-}"
)
if [[ -n "${DURABLE_METADATA_PATH}" ]]; then
  _recovery_cmd+=(--durable-metadata "${DURABLE_METADATA_PATH}")
fi
if ! _resolved_extra="$("${_recovery_cmd[@]}")"; then
  echo "[launch_validation_370m] recovery policy resolution failed" >&2
  exit 2
fi
EXTRA=()
while IFS= read -r _arg; do
  [[ -n "${_arg}" ]] && EXTRA+=("${_arg}")
done <<< "${_resolved_extra}"

echo "[launch_validation_370m] MIX_NAME=${MIX_NAME} DATASET_ID=${DATASET_ID} POOL_DIR=${POOL_DIR:-<auto>} NPROC=${NPROC} TASK_LOSS_NPROC=${TASK_LOSS_NPROC:-${NPROC}} S3_EXPORT=${S3_EXPORT} RECOVERY_MODE=${RECOVERY_MODE} WANDB_MODE=${WANDB_MODE}" >&2

# Ladder task-loss evals use the same GPU count as training (shared across all mix arms).
export TASK_LOSS_NPROC="${TASK_LOSS_NPROC:-${NPROC}}"

if [[ "${NPROC}" -eq 1 ]]; then
  exec "${PYTHON}" "${TRAINER}" "${COMMON_ARGS[@]}" "${EXTRA[@]}"
fi

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  "${TRAINER}" \
  "${COMMON_ARGS[@]}" \
  "${EXTRA[@]}"
