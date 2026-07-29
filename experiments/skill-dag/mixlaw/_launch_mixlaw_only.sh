#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
RSH="ssh -S $SOCK -o BatchMode=yes"
SRC=/mnt/c/alpha_ai/edullm

rsync -avz -e "$RSH" \
  "$SRC/experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh" \
  "$HOST:$STAGING/experiments/skill-dag/mixlaw/"

ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
unset PREFIX || true
export RUN_DIR=/scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236
sed -i 's/\r$//' "$STAGING/experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh"
bash "$STAGING/experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh" | tee /scratch/users/nzhao2/agent-runs/mixlaw-validation-latest-submit.log
echo "=== squeue ==="
squeue --me -o '%.18i %.9P %.30j %.8T %.10M %.6D %R' | head -80
REMOTE
