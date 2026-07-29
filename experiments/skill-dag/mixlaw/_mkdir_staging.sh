#!/usr/bin/env bash
set -Eeuo pipefail
ssh -S /tmp/farmshare-nzhao2.sock -o BatchMode=yes nzhao2@login.farmshare.stanford.edu bash -s <<'REMOTE'
set -Eeuo pipefail
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
mkdir -p \
  "$STAGING/datasets/olmohq" \
  "$STAGING/experiments/skill-dag/mixlaw" \
  "$STAGING/scripts/farmshare" \
  "$STAGING/datasets/olmo" \
  "$STAGING/datasets"
ls -la "$STAGING"
ls "$STAGING/datasets" || true
ls "$STAGING/scripts/farmshare" || true
REMOTE
