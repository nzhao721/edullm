#!/usr/bin/env bash
set -Eeuo pipefail
# Launch olmohq top-up + mixlaw validation on FarmShare.
ssh -S /tmp/farmshare-nzhao2.sock -o BatchMode=yes nzhao2@login.farmshare.stanford.edu bash -s <<'REMOTE'
set -Eeuo pipefail
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
cd "$STAGING"

# Strip CRLF
sed -i 's/\r$//' datasets/olmohq/submit_olmohq_topup.sh experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh

chmod +x datasets/olmohq/submit_olmohq_topup.sh experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh

echo "=== submitting olmohq top-up ==="
bash datasets/olmohq/submit_olmohq_topup.sh | tee /scratch/users/nzhao2/agent-runs/olmohq-topup-latest-submit.log

echo "=== submitting mixlaw validation 10b ==="
bash experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh | tee /scratch/users/nzhao2/agent-runs/mixlaw-validation-latest-submit.log

echo "=== squeue ==="
squeue --me -o '%.18i %.9P %.30j %.8T %.10M %.6D %R' | head -40
REMOTE
