#!/usr/bin/env bash
# Mint AWS session on login; resubmit mixlaw-up (and patch topup-up) without reminting on compute.
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu

ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
MIX=/scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236
TOP=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
EDULLM_ROOT=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"
unset PREFIX || true

# Mint sessions on login for both run dirs.
for RD in "$MIX" "$TOP"; do
  export EDULLM_ROOT RUN_DIR="$RD"
  source "$RD/scripts/prepare_aws_session_light.sh"
  source "$AWS_SESSION_ENV"
  aws sts get-caller-identity --output text
  # Ensure env.sh points at fresh session.
  if [[ "$RD" == "$MIX" ]]; then
    cat > "$MIX/env.sh" <<EOF
RUN_DIR=${MIX}
VENV=${MIX}/venv
EDULLM_ROOT=${EDULLM_ROOT}
AWS_SESSION_ENV=${AWS_SESSION_ENV}
SRC_BUCKET=edullm-datasets
SRC_PREFIX=olmo100b/olmo-mix-1124-30b
DST_BUCKET=edullm-datasets
DST_PREFIX=mixlaw
BUDGET_TOKENS=10000000000
BUILD_WORKERS=4
EOF
  else
    # preserve N/BUCKET/OLMOHQ_PREFIX from existing env if present
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
  fi
done

echo "=== resubmit mixlaw-up (source only; never remint on compute; never write regmix) ==="
UP=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal --cpus-per-task=8 --mem=32G --time=12:00:00 \
  --job-name=mixlaw-up --chdir="$MIX" \
  --output="$MIX/logs/upload-%j.out" --error="$MIX/logs/upload-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; export PATH=\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}; source $MIX/env.sh; source \$AWS_SESSION_ENV; source $MIX/venv/bin/activate; command -v aws; aws sts get-caller-identity --output text; cd $MIX/scripts; python finalize_mixlaw_upload.py --run-dir $MIX --mixtures-json $MIX/plan/validation_mixtures_10b.json --dst-bucket \$DST_BUCKET --dst-prefix \$DST_PREFIX'")
echo "mixlaw_upload_job_id=$UP"
echo "$UP" > "$MIX/upload_job_id.txt"

# Replace pending topup-up if still pending, or if tok almost done attach afterok.
TOK_JOB=$(cat "$TOP/tokenize_job_id.txt")
scancel "$(cat "$TOP/upload_job_id.txt" 2>/dev/null)" 2>/dev/null || true
NEW_UP=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal --cpus-per-task=8 --mem=32G --time=12:00:00 \
  --dependency="afterok:${TOK_JOB}" \
  --job-name=topup-up --chdir="$TOP" \
  --output="$TOP/logs/upload-%j.out" --error="$TOP/logs/upload-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; export PATH=\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}; source $TOP/env.sh; source \$AWS_SESSION_ENV; source $TOP/venv/bin/activate; command -v aws; aws sts get-caller-identity --output text; python $TOP/scripts/finalize_olmohq_topup_upload.py --run-dir $TOP --bucket \$BUCKET --prefix \$OLMOHQ_PREFIX'")
echo "topup_upload_job_id=$NEW_UP"
echo "$NEW_UP" > "$TOP/upload_job_id.txt"

echo "tok progress:"
sacct -j 1666301 --format=JobID,State -n -P | awk -F'|' '$1 ~ /^[0-9]+_[0-9]+$/ {c[$2]++} END {for (s in c) print s, c[s]}'
squeue --me -o '%.12i %.10T %.22j' | head -20
REMOTE
