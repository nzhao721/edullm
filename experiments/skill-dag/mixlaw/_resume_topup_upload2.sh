#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
TOP=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging

# Cancel wasteful re-upload.
ssh -S "$SOCK" -o BatchMode=yes "$HOST" 'scancel 1666961 2>/dev/null || true'

scp -o ControlPath="$SOCK" -o BatchMode=yes \
  /mnt/c/alpha_ai/edullm/datasets/olmohq/finalize_olmohq_topup_upload.py \
  "$HOST:$TOP/scripts/finalize_olmohq_topup_upload.py"
scp -o ControlPath="$SOCK" -o BatchMode=yes \
  /mnt/c/alpha_ai/edullm/datasets/olmohq/finalize_olmohq_topup_upload.py \
  "$HOST:$STAGING/datasets/olmohq/finalize_olmohq_topup_upload.py"

ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
TOP=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
sed -i 's/\r$//' "$TOP/scripts/finalize_olmohq_topup_upload.py"
export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"
unset PREFIX || true
export EDULLM_ROOT=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
export RUN_DIR="$TOP"
source "$TOP/scripts/prepare_aws_session_light.sh"
source "$AWS_SESSION_ENV"
aws sts get-caller-identity --output text
source "$TOP/env.sh" || true
cat > "$TOP/env.sh" <<EOF
RUN_DIR=${TOP}
VENV=${TOP}/venv
EDULLM_ROOT=${EDULLM_ROOT}
AWS_SESSION_ENV=${AWS_SESSION_ENV}
BUCKET=${BUCKET:-edullm-datasets}
OLMOHQ_PREFIX=${OLMOHQ_PREFIX:-olmo100b/olmo-mix-1124-30b}
N=${N:-413}
EOF

UP=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal --cpus-per-task=8 --mem=32G --time=18:00:00 \
  --job-name=topup-up --chdir="$TOP" \
  --output="$TOP/logs/upload-%j.out" --error="$TOP/logs/upload-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; export PATH=\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}; source $TOP/env.sh; source \$AWS_SESSION_ENV; source $TOP/venv/bin/activate; aws sts get-caller-identity --output text; python $TOP/scripts/finalize_olmohq_topup_upload.py --run-dir $TOP --bucket \$BUCKET --prefix \$OLMOHQ_PREFIX'")
echo "topup_upload_job_id=$UP"
echo "$UP" > "$TOP/upload_job_id.txt"
REMOTE
