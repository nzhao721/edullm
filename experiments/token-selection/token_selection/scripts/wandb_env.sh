# Shared W&B enablement for token-selection arm launches (SmolLM2 protocol).
# Source from arm launch_*.sh before exec'ing the trainer:
#   # shellcheck source=../token_selection/scripts/wandb_env.sh
#   source "${TS_ROOT}/token_selection/scripts/wandb_env.sh" [arm] [run_name]
#
# Optional env:
#   WANDB_SESSION_ENV / RUN_DIR/wandb-session.env — provides WANDB_API_KEY
#   WANDB_PROJECT (default: token-selection)
#   WANDB_MODE (default: online; set disabled to skip)
#   WANDB_ENTITY, WANDB_RUN_NAME, WANDB_GROUP, WANDB_UPLOAD_EXISTING
#
# Does not weaken S3 / edullm-data fail-closed; W&B is additive logging.
: "${WANDB_PROJECT:=token-selection}"
: "${WANDB_MODE:=online}"

_ARM_HINT="${1:-}"
_RUN_HINT="${2:-}"

if [[ -n "${WANDB_SESSION_ENV:-}" && -f "${WANDB_SESSION_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${WANDB_SESSION_ENV}"
elif [[ -n "${RUN_DIR:-}" && -f "${RUN_DIR}/wandb-session.env" ]]; then
  # shellcheck disable=SC1091
  source "${RUN_DIR}/wandb-session.env"
fi

if [[ -n "${_RUN_HINT}" ]]; then
  : "${WANDB_RUN_NAME:=${_RUN_HINT}}"
  : "${WANDB_NAME:=${_RUN_HINT}}"
fi
if [[ -n "${_ARM_HINT}" ]]; then
  : "${WANDB_GROUP:=${_ARM_HINT}}"
fi

# Soft-enable: never hard-disable when mode is online/offline (trainer soft-skips
# without WANDB_API_KEY). Explicit WANDB_MODE=disabled keeps W&B off.
if [[ "${WANDB_MODE}" == "disabled" ]]; then
  export WANDB_DISABLED=1
else
  unset WANDB_DISABLED || true
fi

export WANDB_PROJECT WANDB_MODE
if [[ -n "${WANDB_ENTITY:-}" ]]; then export WANDB_ENTITY; fi
if [[ -n "${WANDB_RUN_NAME:-}" ]]; then export WANDB_RUN_NAME WANDB_NAME; fi
if [[ -n "${WANDB_GROUP:-}" ]]; then export WANDB_GROUP; fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then export WANDB_API_KEY; fi
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "warn: WANDB_MODE=online but WANDB_API_KEY unset; trainer will continue without W&B" >&2
  echo "      push wandb-session.env via scripts/farmshare/push_wandb_session_to_farmshare.sh" >&2
fi
