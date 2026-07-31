#!/usr/bin/env bash
# Refresh AWS_SESSION_ENV from the FarmShare login node every INTERVAL seconds.
# Compute nodes cannot remint (broker needs Node); login can. Scratch is shared.
set -Eeuo pipefail

: "${RUN_DIR:?}"
INTERVAL="${INTERVAL:-1200}"
SUNET="${SUNET:-nzhao2}"
STAGING_ROOT="${STAGING_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
WRITER="${WRITER:-${STAGING_ROOT}/scripts/farmshare/write_aws_session_env.py}"
OUT="${RUN_DIR}/aws-session.env"
LOG="${RUN_DIR}/logs/aws_session_refresh.log"
PROFILE="${AWS_PROFILE:-sbsandbox}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

mkdir -p "${RUN_DIR}/logs"
PY="${RUN_DIR}/venv/bin/python"
if [[ ! -x "$PY" ]]; then PY=python3; fi

# credential_process needs node via nvm on FarmShare login nodes
if [[ -s "${HOME}/.nvm/nvm.sh" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.nvm/nvm.sh"
fi
export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"

echo "[$(date -u +%FT%TZ)] refresher start interval=${INTERVAL}s out=${OUT}" >>"$LOG"
while true; do
  if [[ -f "${RUN_DIR}/STOP_AWS_REFRESH" ]]; then
    echo "[$(date -u +%FT%TZ)] stop file seen; exiting" >>"$LOG"
    exit 0
  fi
# Prefer chaining from the current session file so we don't need sb-aws login each time.
if [[ -f "$OUT" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$OUT"
  set -u
fi
  if "$PY" "$WRITER" --output "$OUT" --profile "$PROFILE" --region "$REGION" --force-new \
      >>"$LOG" 2>&1; then
    chmod 600 "$OUT"
    # shellcheck disable=SC1090
    set +u
    source "$OUT"
    set -u
    if aws sts get-caller-identity --query Arn --output text >>"$LOG" 2>&1; then
      echo "[$(date -u +%FT%TZ)] refreshed ok" >>"$LOG"
    else
      echo "[$(date -u +%FT%TZ)] refreshed file but sts failed" >>"$LOG"
    fi
  else
    echo "[$(date -u +%FT%TZ)] refresh FAILED" >>"$LOG"
  fi
  sleep "$INTERVAL"
done
