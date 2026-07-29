#!/usr/bin/env bash
# Resubmit mixlaw pool→slice→upload with aws on PATH. Does not touch regmix-10b.
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu

ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
MIX=/scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236
TOP=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841

# Cancel dead/pending mixlaw dependents from prior failed pool.
for j in 1666304 1666305; do
  scancel "$j" 2>/dev/null || true
done

# Refresh AWS session on login so pool has credentials.
unset PREFIX || true
export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"
export EDULLM_ROOT=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
export RUN_DIR="$MIX"
source "$MIX/scripts/prepare_aws_session_light.sh"
source "$AWS_SESSION_ENV"
# Rewrite env.sh with fresh session path.
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

echo "=== topup progress (tok must finish before we *need* refreshed manifest; pool can use current) ==="
sacct -j 1666301 --format=State -n -P | sort | uniq -c || true
echo "topup upload still pending on 1666302"

echo "resubmitting mixlaw pool/slice/up with PATH+aws"
POOL_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal --cpus-per-task=8 --mem=64G --time=12:00:00 \
  --job-name=mixlaw-pool --chdir="$MIX" \
  --output="$MIX/logs/pool-%j.out" --error="$MIX/logs/pool-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; export PATH=\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}; source $MIX/env.sh; source \$AWS_SESSION_ENV; source $MIX/venv/bin/activate; command -v aws; cd $MIX/scripts; python build_working_pool_from_shards.py --tokenized-manifest $MIX/plan/tokenized_manifest.json --s3-tokenized-prefix s3://\$SRC_BUCKET/\$SRC_PREFIX/tokenized --out-dir $MIX/pool --mixtures-json $MIX/plan/validation_mixtures_10b.json --budget-tokens \$BUDGET_TOKENS'")
echo "pool_job_id=$POOL_JOB"

SLICE_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal --cpus-per-task=16 --mem=64G --time=12:00:00 \
  --dependency="afterok:${POOL_JOB}" \
  --job-name=mixlaw-slice --chdir="$MIX" \
  --output="$MIX/logs/slice-%j.out" --error="$MIX/logs/slice-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; source $MIX/env.sh; source $MIX/venv/bin/activate; cd $MIX/scripts; python build_mixture_data.py plan --tokenized-dir $MIX/pool/tokenized --out-dir $MIX/slices --mixtures-json $MIX/plan/validation_mixtures_10b.json --total-tokens \$BUDGET_TOKENS; python build_mixture_data.py build --plan-dir $MIX/slices --out-dir $MIX/slices --workers \$BUILD_WORKERS'")
echo "slice_job_id=$SLICE_JOB"

UP2=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal --cpus-per-task=8 --mem=32G --time=12:00:00 \
  --dependency="afterok:${SLICE_JOB}" \
  --job-name=mixlaw-up --chdir="$MIX" \
  --output="$MIX/logs/upload-%j.out" --error="$MIX/logs/upload-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; export PATH=\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}; source $MIX/env.sh; export EDULLM_ROOT=\$EDULLM_ROOT RUN_DIR=$MIX; source $MIX/scripts/prepare_aws_session_light.sh; source \$AWS_SESSION_ENV; source $MIX/venv/bin/activate; cd $MIX/scripts; python finalize_mixlaw_upload.py --run-dir $MIX --mixtures-json $MIX/plan/validation_mixtures_10b.json --dst-bucket \$DST_BUCKET --dst-prefix \$DST_PREFIX'")
echo "upload_job_id=$UP2"
echo "$POOL_JOB" > "$MIX/pool_job_id.txt"
echo "$SLICE_JOB" > "$MIX/slice_job_id.txt"
echo "$UP2" > "$MIX/upload_job_id.txt"

# Also patch pending topup-up wrap if still pending — cancel and reattach after tok.
UP_PENDING=$(squeue -j 1666302 -h -o '%T' 2>/dev/null || true)
if [[ "$UP_PENDING" == "PENDING" ]]; then
  echo "replacing pending topup-up 1666302 with PATH-safe wrap"
  scancel 1666302
  TOK_JOB=$(cat "$TOP/tokenize_job_id.txt")
  NEW_UP=$(sbatch --parsable --exclude=wheat-01 \
    --partition=normal --cpus-per-task=8 --mem=32G --time=12:00:00 \
    --dependency="afterok:${TOK_JOB}" \
    --job-name=topup-up --chdir="$TOP" \
    --output="$TOP/logs/upload-%j.out" --error="$TOP/logs/upload-%j.err" \
    --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; export PATH=\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}; source $TOP/env.sh; export EDULLM_ROOT=\$EDULLM_ROOT RUN_DIR=$TOP; source $TOP/scripts/prepare_aws_session_light.sh; source \$AWS_SESSION_ENV; source $TOP/venv/bin/activate; python $TOP/scripts/finalize_olmohq_topup_upload.py --run-dir $TOP --bucket \$BUCKET --prefix \$OLMOHQ_PREFIX'")
  echo "new_topup_upload_job_id=$NEW_UP"
  echo "$NEW_UP" > "$TOP/upload_job_id.txt"
fi

squeue --me -o '%.12i %.10T %.22j' | head -40
REMOTE
