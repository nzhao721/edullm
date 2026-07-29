#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu

ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
TOP=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
MIX=/scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236
unset PREFIX || true
source "$TOP/env.sh"
N=$(wc -l < "$TOP/plan/topup_manifest.jsonl" | tr -d ' ')
echo "resubmitting topup map/tok/up N=$N (downloads already complete)"

# Downloads done — no dependency on DL job.
MAP_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal --cpus-per-task=2 --mem=4G --time=00:30:00 \
  --job-name=topup-map \
  --chdir="$TOP" \
  --output="$TOP/logs/map-%j.out" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; source $TOP/env.sh; source $TOP/venv/bin/activate; export RUN_DIR=$TOP; python $TOP/scripts/build_topup_tokenize_map.py'")
echo "map_job_id=$MAP_JOB"

TOK_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --array=0-$((N - 1))%20 \
  --dependency="afterok:${MAP_JOB}" \
  --chdir="$TOP" \
  --output="$TOP/logs/tok-%A_%a.out" \
  --error="$TOP/logs/tok-%A_%a.err" \
  --export=ALL,RUN_DIR=$TOP \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; source $TOP/env.sh; source $TOP/venv/bin/activate; LINE=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID + 1))p\" $TOP/tokenize_map.txt); SRC=\${LINE%%|*}; DST=\${LINE##*|}; mkdir -p \"\$(dirname \"\$DST\")\"; $TOP/venv/bin/python $TOP/scripts/tokenize_olmo_shard.py --input \"\$SRC\" --output \"\$DST\"'")
echo "tokenize_job_id=$TOK_JOB"

UP_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal --cpus-per-task=8 --mem=32G --time=12:00:00 \
  --dependency="afterok:${TOK_JOB}" \
  --job-name=topup-up \
  --chdir="$TOP" \
  --output="$TOP/logs/upload-%j.out" \
  --error="$TOP/logs/upload-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; source $TOP/env.sh; export EDULLM_ROOT=$EDULLM_ROOT RUN_DIR=$TOP; source $TOP/scripts/prepare_aws_session_light.sh; source \$AWS_SESSION_ENV; source $TOP/venv/bin/activate; python $TOP/scripts/finalize_olmohq_topup_upload.py --run-dir $TOP --bucket $BUCKET --prefix \$OLMOHQ_PREFIX'")
echo "upload_job_id=$UP_JOB"
echo "$MAP_JOB" > "$TOP/map_job_id.txt"
echo "$TOK_JOB" > "$TOP/tokenize_job_id.txt"
echo "$UP_JOB" > "$TOP/upload_job_id.txt"

source "$MIX/env.sh"
echo "resubmitting mixlaw pool/slice/up"
POOL_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal --cpus-per-task=8 --mem=64G --time=12:00:00 \
  --job-name=mixlaw-pool --chdir="$MIX" \
  --output="$MIX/logs/pool-%j.out" --error="$MIX/logs/pool-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; source $MIX/env.sh; source \$AWS_SESSION_ENV; source $MIX/venv/bin/activate; cd $MIX/scripts; python build_working_pool_from_shards.py --tokenized-manifest $MIX/plan/tokenized_manifest.json --s3-tokenized-prefix s3://\$SRC_BUCKET/\$SRC_PREFIX/tokenized --out-dir $MIX/pool --mixtures-json $MIX/plan/validation_mixtures_10b.json --budget-tokens \$BUDGET_TOKENS'")
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
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; source $MIX/env.sh; export EDULLM_ROOT=\$EDULLM_ROOT RUN_DIR=$MIX; source $MIX/scripts/prepare_aws_session_light.sh; source \$AWS_SESSION_ENV; source $MIX/venv/bin/activate; cd $MIX/scripts; python finalize_mixlaw_upload.py --run-dir $MIX --mixtures-json $MIX/plan/validation_mixtures_10b.json --dst-bucket \$DST_BUCKET --dst-prefix \$DST_PREFIX'")
echo "upload_job_id=$UP2"
echo "$POOL_JOB" > "$MIX/pool_job_id.txt"
echo "$SLICE_JOB" > "$MIX/slice_job_id.txt"
echo "$UP2" > "$MIX/upload_job_id.txt"

squeue --me -o '%.12i %.10T %.22j' | head -40
REMOTE
