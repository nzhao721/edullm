#!/usr/bin/env bash
# Lightweight AWS session mint for FarmShare compute nodes (no Dolma/HF deps).
set -Eeuo pipefail

: "${EDULLM_ROOT:?Set EDULLM_ROOT}"
: "${RUN_DIR:?Set RUN_DIR}"
: "${AWS_DEFAULT_REGION:=us-east-1}"

SCRIPT_DIR="${RUN_DIR}/scripts"

credential_profile="${AWS_PROFILE:-sbsandbox}"
unset AWS_SESSION_ENV
export AWS_PROFILE="${credential_profile}" AWS_DEFAULT_REGION
export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"

if [[ -s "${HOME}/.nvm/nvm.sh" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.nvm/nvm.sh"
fi

command -v aws >/dev/null
command -v python3 >/dev/null

mkdir -p "${RUN_DIR}"
chmod 700 "${RUN_DIR}"
AWS_SESSION_ENV="${RUN_DIR}/aws-session.env"

# Prefer run-dir venv python (has boto3); fall back to system python3.
PY="${RUN_DIR}/venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

"${PY}" "${SCRIPT_DIR}/write_aws_session_env.py" \
  --output "${AWS_SESSION_ENV}" \
  --profile "${credential_profile}" \
  --region "${AWS_DEFAULT_REGION}"
chmod 600 "${AWS_SESSION_ENV}"
export AWS_SESSION_ENV

# shellcheck disable=SC1090
source "${AWS_SESSION_ENV}"
aws sts get-caller-identity --output text >/dev/null
echo "aws_session_ready profile=${credential_profile} file=${AWS_SESSION_ENV}"
