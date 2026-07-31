#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT=/mnt/c/alpha_ai/edullm/scripts/farmshare/loop_push_aws_session_to_farmshare.sh
LOG=/tmp/farmshare-aws-session-push.log
PIDFILE=/tmp/farmshare-aws-session-push.pid
RUNS=(
  /scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z
)

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PIDFILE")"
  exit 0
fi

pkill -f "loop_push_aws_session_to_farmshare.sh" 2>/dev/null || true
sleep 1

nohup env INTERVAL=600 bash "$SCRIPT" "${RUNS[@]}" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "started pid=$(cat "$PIDFILE") log=$LOG interval=600s"
sleep 2
tail -n 10 "$LOG" || true
