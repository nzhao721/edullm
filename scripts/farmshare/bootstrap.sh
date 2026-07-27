#!/usr/bin/env bash
# Shared FarmShare Slurm environment. Secrets are read from user-owned stores,
# never from repository files or Slurm exports.
set -Eeuo pipefail

: "${EDULLM_ROOT:?Set EDULLM_ROOT to the uploaded repository path}"
: "${DOLMA_VENV:=${HOME}/.venvs/edullm-dolma}"
: "${AWS_DEFAULT_REGION:=us-east-1}"

if [[ -n "${AWS_SESSION_ENV:-}" ]]; then
  if [[ ! -r "${AWS_SESSION_ENV}" ]]; then
    echo "missing readable AWS session file: ${AWS_SESSION_ENV}" >&2
    exit 1
  fi
  # The session file is generated on the login node from sbsandbox credentials.
  # shellcheck disable=SC1090
  source "${AWS_SESSION_ENV}"
  : "${AWS_ACCESS_KEY_ID:?AWS session file has no access key}"
  : "${AWS_SECRET_ACCESS_KEY:?AWS session file has no secret key}"
  : "${AWS_SESSION_TOKEN:?AWS session file has no session token}"
  # Explicit temporary credentials must take precedence over credential_process.
  unset AWS_PROFILE
else
  : "${AWS_PROFILE:=sbsandbox}"
  export AWS_PROFILE
fi

if [[ -s "${HOME}/.nvm/nvm.sh" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.nvm/nvm.sh"
fi
export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"
export AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION}}"
export AWS_DEFAULT_REGION PYTHONUNBUFFERED=1

if [[ ! -x "${DOLMA_VENV}/bin/python" ]]; then
  echo "missing Dolma virtualenv: ${DOLMA_VENV}; run scripts/farmshare/setup_runtime.sh" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${DOLMA_VENV}/bin/activate"

if [[ -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(<"${HOME}/.cache/huggingface/token")"
elif [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "missing Hugging Face token in ~/.cache/huggingface/token" >&2
  exit 1
fi

command -v aws >/dev/null
command -v dolma >/dev/null
aws sts get-caller-identity --output json >/dev/null

cd "${EDULLM_ROOT}"
