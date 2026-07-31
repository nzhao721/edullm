#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
REPO=/mnt/c/alpha_ai/edullm
RUN=/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z

echo "=== socket check ==="
if ssh -S "$SOCK" -O check "$HOST" 2>&1; then
  echo socket_ok
else
  echo "SOCKET DEAD — need student to reopen ControlMaster"
  exit 2
fi

echo "=== push fresh session ==="
bash "$REPO/scripts/farmshare/push_aws_session_to_farmshare.sh" "$RUN"

echo "=== restart local pusher ==="
pkill -f loop_push_aws_session_to_farmshare.sh 2>/dev/null || true
sleep 1
rm -f /tmp/farmshare-aws-session-push.pid
bash "$REPO/scripts/farmshare/_start_local_pusher.sh"

echo "=== job + logs ==="
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<EOF
set -e
JOB=1670744
sacct -j \$JOB --format=JobID,State,ExitCode,Elapsed,NodeList,ReqMem,MaxRSS -P | head -5
squeue -j \$JOB -o '%.18i %.2t %.10M %R' 2>/dev/null || true
echo ---out---
wc -c $RUN/logs/olmo127b_edullm_publish_\$JOB.out $RUN/logs/olmo127b_edullm_publish_\$JOB.e 2>&1 || true
tail -60 $RUN/logs/olmo127b_edullm_publish_\$JOB.out 2>&1 || true
echo ---err---
tr -d '\\000' < $RUN/logs/olmo127b_edullm_publish_\$JOB.e 2>/dev/null | tail -80 || true
echo ---token_errors---
grep -E 'ExpiredToken|expired|Credential|AccessDenied|Error|Traceback|rotate' \
  $RUN/logs/olmo127b_edullm_publish_\$JOB.out \
  $RUN/logs/olmo127b_edullm_publish_\$JOB.e 2>/dev/null | tail -30 || echo none
EOF
