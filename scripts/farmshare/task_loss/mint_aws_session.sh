#!/usr/bin/env bash
# Mint temporary AWS session credentials onto FarmShare without printing secrets.
#
# Preferred: resolve via FarmShare's sbsandbox profile / sb-aws-creds.
# Fallback: if LOCAL_SESSION_ENV is set to a private env file already present
# on the login node (copied over ControlMaster without echoing), install it.
set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${EDULLM_ROOT:=${HOME}/edullm}"
: "${AWS_DEFAULT_REGION:=us-east-1}"
: "${AWS_PROFILE:=sbsandbox}"

mkdir -p "${RUN_DIR}"
chmod 700 "${RUN_DIR}"
OUT="${RUN_DIR}/aws-session.env"

if [[ -n "${LOCAL_SESSION_ENV:-}" && -f "${LOCAL_SESSION_ENV}" ]]; then
  cp -f "${LOCAL_SESSION_ENV}" "${OUT}"
  chmod 600 "${OUT}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  WRITER=""
  for cand in \
    "${SCRIPT_DIR}/write_aws_session_env.py" \
    "${EDULLM_ROOT}/scripts/farmshare/write_aws_session_env.py" \
    "${SCRIPT_DIR}/../write_aws_session_env.py"
  do
    if [[ -f "${cand}" ]]; then
      WRITER="${cand}"
      break
    fi
  done
  if [[ -z "${WRITER}" ]]; then
    echo "missing write_aws_session_env.py; set LOCAL_SESSION_ENV" >&2
    exit 1
  fi
  export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"
  if [[ -s "${HOME}/.nvm/nvm.sh" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.nvm/nvm.sh"
  fi
  # Prefer a venv python that has boto3.
  PY=python3
  for cand in \
    "${RUN_DIR}/venv/bin/python" \
    "/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/venv/bin/python" \
    "${HOME}/.venvs/edullm-dolma/bin/python"
  do
    if [[ -x "${cand}" ]]; then
      PY="${cand}"
      break
    fi
  done
  unset AWS_SESSION_ENV AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
  export AWS_PROFILE AWS_DEFAULT_REGION
  "${PY}" "${WRITER}" --output "${OUT}" --profile "${AWS_PROFILE}" --region "${AWS_DEFAULT_REGION}"
  chmod 600 "${OUT}"
fi

# Verify without printing credential values.
# shellcheck disable=SC1090
source "${OUT}"
unset AWS_PROFILE
aws sts get-caller-identity --output text >/dev/null
echo "aws_session_ready file=${OUT} account=$(aws sts get-caller-identity --query Account --output text) arn=$(aws sts get-caller-identity --query Arn --output text)"
export AWS_SESSION_ENV="${OUT}"
