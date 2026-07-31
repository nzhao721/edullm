#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
RUN=/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z
JOB=1670744

echo "=== local pusher ==="
pgrep -af loop_push_aws_session || echo 'no loop_push'
tail -12 /tmp/farmshare-aws-session-push.log 2>/dev/null || true

ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<EOF
set -e
echo "=== job ==="
sacct -j $JOB --format=JobID,State,ExitCode,Elapsed,NodeList,ReqMem,MaxRSS -P | head -8
squeue -j $JOB -o '%.18i %.2t %.10M %R' 2>/dev/null || true
sstat -j ${JOB}.batch --format=JobID,MaxRSS,AveRSS,AveCPU 2>/dev/null || true
echo "=== out ==="
tail -80 $RUN/logs/olmo127b_edullm_publish_${JOB}.out 2>&1 || true
echo "=== err ==="
tr -d '\\000' < $RUN/logs/olmo127b_edullm_publish_${JOB}.e 2>/dev/null | tail -60 || true
echo "=== errors ==="
grep -E 'ExpiredToken|Error|Traceback|OOM|Killed|done|publish|rotate|upload|hash' \
  $RUN/logs/olmo127b_edullm_publish_${JOB}.out \
  $RUN/logs/olmo127b_edullm_publish_${JOB}.e 2>/dev/null | tail -40 || echo none
echo "=== session mtime ==="
stat -c 'mtime=%y' $RUN/aws-session.env 2>/dev/null || true
grep AWS_SESSION_EXPIRATION $RUN/aws-session.env 2>/dev/null || true
EOF
