#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
RSH="ssh -S $SOCK -o BatchMode=yes"
SRC=/mnt/c/alpha_ai/edullm

rsync -avz -e "$RSH" \
  "$SRC/datasets/olmohq/submit_olmohq_topup.sh" \
  "$SRC/datasets/olmohq/plan_olmohq_topup.py" \
  "$SRC/datasets/olmohq/finalize_olmohq_topup_upload.py" \
  "$HOST:$STAGING/datasets/olmohq/"

rsync -avz -e "$RSH" \
  "$SRC/experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh" \
  "$HOST:$STAGING/experiments/skill-dag/mixlaw/"

ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
sed -i 's/\r$//' "$STAGING/datasets/olmohq/"*.sh "$STAGING/datasets/olmohq/"*.py "$STAGING/experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh"
chmod +x "$STAGING/datasets/olmohq/submit_olmohq_topup.sh" "$STAGING/experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh"
unset PREFIX || true

# Reuse prior run dir + venv to skip pip
export RUN_DIR=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
echo "=== olmohq top-up RUN_DIR=$RUN_DIR ==="
bash "$STAGING/datasets/olmohq/submit_olmohq_topup.sh" | tee /scratch/users/nzhao2/agent-runs/olmohq-topup-latest-submit.log

echo "=== mixlaw validation ==="
unset PREFIX || true
bash "$STAGING/experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh" | tee /scratch/users/nzhao2/agent-runs/mixlaw-validation-latest-submit.log

squeue --me -o '%.18i %.9P %.30j %.8T %.10M %.6D %R' | head -60
REMOTE
