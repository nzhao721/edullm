#!/usr/bin/env bash
# Source on a FarmShare login node to prepare compute-node AWS credentials.
set -Eeuo pipefail

: "${EDULLM_ROOT:?Set EDULLM_ROOT to the uploaded repository path}"
: "${RUN_DIR:?Set RUN_DIR to the isolated smoke-run directory}"
: "${AWS_DEFAULT_REGION:=us-east-1}"

# Resolve the broker-backed profile on the login node.  A stale exported session
# path must never cause this script to skip that login-node resolution.
credential_profile="${AWS_PROFILE:-sbsandbox}"
unset AWS_SESSION_ENV
export AWS_PROFILE="${credential_profile}" AWS_DEFAULT_REGION

# shellcheck source=bootstrap.sh
source "${EDULLM_ROOT}/scripts/farmshare/bootstrap.sh"

mkdir -p "${RUN_DIR}"
chmod 700 "${RUN_DIR}"
AWS_SESSION_ENV="${RUN_DIR}/aws-session.env"
python "${EDULLM_ROOT}/scripts/farmshare/write_aws_session_env.py" \
  --output "${AWS_SESSION_ENV}" \
  --profile "${credential_profile}" \
  --region "${AWS_DEFAULT_REGION}"
chmod 600 "${AWS_SESSION_ENV}"
export AWS_SESSION_ENV
