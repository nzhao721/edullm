#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
RUN=/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<EOF
set -e
echo "=== session file ==="
# show expiry only, not secrets
grep -E 'AWS_SESSION_EXPIRATION|AWS_ACCESS_KEY_ID' $RUN/aws-session.env | sed 's/AWS_ACCESS_KEY_ID=.*/AWS_ACCESS_KEY_ID=.../' 
stat -c 'mtime=%y size=%s' $RUN/aws-session.env
echo "=== process on node ==="
scontrol listpids 1670744 2>/dev/null | head -20 || true
# sample RSS of python if visible via sstat
sstat -j 1670744.batch --format=JobID,MaxRSS,AveRSS,MaxVMSize,AveCPU 2>/dev/null || true
echo "=== landing presence (list top) ==="
source $RUN/aws-session.env
export PATH="\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}"
aws s3 ls s3://edullm-landing/pretrain/olmo-127b/ 2>&1 | head -20 || true
aws s3 ls s3://edullm-landing/pretrain/olmo-127b/v1/ 2>&1 | head -30 || true
echo "=== node disk / stage ==="
du -sh $RUN/publish-stage 2>/dev/null || true
ls $RUN/publish-stage/tokens 2>/dev/null | head
EOF
