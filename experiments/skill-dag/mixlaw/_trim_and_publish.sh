#!/usr/bin/env bash
# Trim active olmohq manifests to |plan-meas|/meas<=10%; republish (regmix untouched).
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
TOP=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging

scp -o ControlPath="$SOCK" -o BatchMode=yes \
  /mnt/c/alpha_ai/edullm/datasets/olmohq/trim_olmohq_topup_manifest.py \
  "$HOST:$TOP/scripts/trim_olmohq_topup_manifest.py"
scp -o ControlPath="$SOCK" -o BatchMode=yes \
  /mnt/c/alpha_ai/edullm/datasets/olmohq/trim_olmohq_topup_manifest.py \
  "$HOST:$STAGING/datasets/olmohq/trim_olmohq_topup_manifest.py"

ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
TOP=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
sed -i 's/\r$//' "$TOP/scripts/trim_olmohq_topup_manifest.py"
export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"
unset PREFIX || true
export EDULLM_ROOT=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
export RUN_DIR="$TOP"
source "$TOP/scripts/prepare_aws_session_light.sh"
source "$AWS_SESSION_ENV"
source "$TOP/venv/bin/activate"
source "$TOP/env.sh"

python "$TOP/scripts/trim_olmohq_topup_manifest.py" --run-dir "$TOP"

# Publish trimmed manifests as the active inventory (backup overshoot first).
ROOT="s3://${BUCKET}/${OLMOHQ_PREFIX}"
aws s3 cp "$TOP/plan/tokenized_manifest_merged.json" "$ROOT/plan/tokenized_manifest.overshoot.json" --only-show-errors
aws s3 cp "$TOP/plan/manifest_merged.jsonl" "$ROOT/plan/manifest.overshoot.jsonl" --only-show-errors
aws s3 cp "$TOP/plan/tokenized_manifest_trimmed.json" "$ROOT/plan/tokenized_manifest.json" --only-show-errors
aws s3 cp "$TOP/plan/manifest_trimmed.jsonl" "$ROOT/plan/manifest.jsonl" --only-show-errors
aws s3 cp "$TOP/plan/availability_after_topup.json" "$ROOT/plan/availability_after_topup.json" --only-show-errors
echo "trimmed manifests published; regmix untouched"
cat "$TOP/plan/availability_after_topup.json"
REMOTE
