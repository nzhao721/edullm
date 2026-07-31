#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
RUN=/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<EOF
set -e
JOB=1670744
sacct -j \$JOB --format=JobID,State,ExitCode,Elapsed,NodeList,ReqMem,MaxRSS -P | head -5
squeue -j \$JOB -o '%.18i %.2t %.10M %R %m' 2>/dev/null || true
echo ---out---
tail -50 $RUN/logs/olmo127b_edullm_publish_\$JOB.out 2>&1 || true
echo ---err---
tr -d '\\000' < $RUN/logs/olmo127b_edullm_publish_\$JOB.e 2>/dev/null | tail -50 || true
EOF
