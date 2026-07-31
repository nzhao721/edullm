#!/usr/bin/env bash
# Periodically mint laptop AWS credentials and push to FarmShare run dirs.
# Prefer this over FarmShare-side sb-aws-creds login (which does not work).
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL="${INTERVAL:-1200}"
LOG="${LOG:-/tmp/farmshare-aws-session-push.log}"

if [[ $# -lt 1 ]]; then
  echo "usage: INTERVAL=1200 $0 RUN_DIR [RUN_DIR...]" >&2
  exit 2
fi

echo "[$(date -u +%FT%TZ)] local_pusher start interval=${INTERVAL}s runs=$*" | tee -a "$LOG"
while true; do
  if bash "$SCRIPT_DIR/push_aws_session_to_farmshare.sh" "$@" >>"$LOG" 2>&1; then
    echo "[$(date -u +%FT%TZ)] push ok" | tee -a "$LOG"
  else
    echo "[$(date -u +%FT%TZ)] push FAILED (is Windows sb-aws-creds logged in?)" | tee -a "$LOG"
  fi
  sleep "$INTERVAL"
done
