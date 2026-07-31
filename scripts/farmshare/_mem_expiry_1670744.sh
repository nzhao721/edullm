#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
RUN=/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<EOF
set -e
# expiry line only
grep -E 'EXPIRATION|expiration' $RUN/aws-session.env | sed 's/=.*/=<redacted-or-shown>/' 
# actually show expiration value — it's not a secret
grep -E 'AWS_SESSION_EXPIRATION|EXPIR' $RUN/aws-session.env || true
echo "=== dmesg/oom? ==="
# can't read other users dmesg; check job still running and exit
sacct -j 1670744 --format=JobID,State,ExitCode,Elapsed,ReqMem,MaxRSS -P | head -5
echo "=== log sizes over time ==="
ls -l $RUN/logs/olmo127b_edullm_publish_1670744.*
# any temp under workdir?
ls $RUN | head -30
# python open files count via /proc if we can get pid
PID=\$(scontrol listpids 1670744 2>/dev/null | awk 'NR==2{print \$2}')
echo "batch_pid=\$PID"
if [[ -n "\$PID" && -r /proc/\$PID/status ]]; then
  grep -E 'VmRSS|VmSize|Threads' /proc/\$PID/status || true
fi
# On compute node we can't see /proc from login - listpids is from controller
echo "=== recent err null-stripped full ==="
tr -d '\\000' < $RUN/logs/olmo127b_edullm_publish_1670744.e | tail -100
EOF
