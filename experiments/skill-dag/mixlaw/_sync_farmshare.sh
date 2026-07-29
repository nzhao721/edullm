#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
RSH="ssh -S $SOCK -o BatchMode=yes"
SRC=/mnt/c/alpha_ai/edullm

rsync -avz -e "$RSH" \
  "$SRC/datasets/olmohq/plan_olmohq_topup.py" \
  "$SRC/datasets/olmohq/finalize_olmohq_topup_upload.py" \
  "$SRC/datasets/olmohq/submit_olmohq_topup.sh" \
  "$HOST:$STAGING/datasets/olmohq/"

rsync -avz -e "$RSH" \
  "$SRC/experiments/skill-dag/mixlaw/mixlaw_common.py" \
  "$SRC/experiments/skill-dag/mixlaw/build_mixture_data.py" \
  "$SRC/experiments/skill-dag/mixlaw/build_working_pool_from_shards.py" \
  "$SRC/experiments/skill-dag/mixlaw/finalize_mixlaw_upload.py" \
  "$SRC/experiments/skill-dag/mixlaw/write_validation_mixtures.py" \
  "$SRC/experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh" \
  "$SRC/experiments/skill-dag/mixlaw/validation_mixtures_10b.json" \
  "$SRC/experiments/skill-dag/mixlaw/mixlaw_fit_chinchilla.json" \
  "$SRC/experiments/skill-dag/mixlaw/mixlaw_fit_lightgbm_chinchilla.json" \
  "$SRC/experiments/skill-dag/mixlaw/mixtures.json" \
  "$HOST:$STAGING/experiments/skill-dag/mixlaw/"

# Ensure farmshare helpers exist on remote
ssh -S "$SOCK" -o BatchMode=yes "$HOST" "ls $STAGING/scripts/farmshare/prepare_aws_session_light.sh $STAGING/scripts/farmshare/write_aws_session_env.py"
echo SYNC_OK
