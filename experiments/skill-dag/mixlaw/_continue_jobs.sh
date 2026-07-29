#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
RSH="ssh -S $SOCK -o BatchMode=yes"
SRC=/mnt/c/alpha_ai/edullm

rsync -avz -e "$RSH" \
  "$SRC/datasets/olmohq/finalize_olmohq_topup_upload.py" \
  "$SRC/datasets/olmohq/submit_olmohq_topup.sh" \
  "$HOST:$STAGING/datasets/olmohq/"

rsync -avz -e "$RSH" \
  "$SRC/experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh" \
  "$HOST:$STAGING/experiments/skill-dag/mixlaw/"

ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
RUN_DIR=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
unset PREFIX || true

# Ensure env.sh has OLMOHQ_PREFIX
if ! grep -q OLMOHQ_PREFIX "${RUN_DIR}/env.sh"; then
  echo "OLMOHQ_PREFIX=olmo100b/olmo-mix-1124-30b" >> "${RUN_DIR}/env.sh"
fi
source "${RUN_DIR}/env.sh"
N=$(wc -l < "${RUN_DIR}/plan/topup_manifest.jsonl" | tr -d ' ')
DL_JOB=$(cat "${RUN_DIR}/download_job_id.txt")
echo "continuing from DL_JOB=${DL_JOB} N=${N}"

cp -a "$STAGING/datasets/olmohq/finalize_olmohq_topup_upload.py" "${RUN_DIR}/scripts/"
sed -i 's/\r$//' "${RUN_DIR}/scripts/"*.py "${RUN_DIR}/scripts/"*.sbatch || true

# Patch tokenize sbatch to use venv python absolute path
sed -i 's|python "${RUN_DIR}/scripts/tokenize_olmo_shard.py"|'"${RUN_DIR}"'/venv/bin/python "${RUN_DIR}/scripts/tokenize_olmo_shard.py"|' \
  "${RUN_DIR}/scripts/tokenize_topup.sbatch" || true

MAP_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal --cpus-per-task=2 --mem=4G --time=00:30:00 \
  --dependency="afterok:${DL_JOB}" \
  --job-name=topup-map \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/map-%j.out" \
  --wrap="set -Eeuo pipefail; unset PREFIX || true; source ${RUN_DIR}/env.sh; source ${RUN_DIR}/venv/bin/activate; export RUN_DIR=${RUN_DIR}; python ${RUN_DIR}/scripts/build_topup_tokenize_map.py")
echo "map_job_id=${MAP_JOB}"

TOK_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --array=0-$((N - 1))%20 \
  --dependency="afterok:${MAP_JOB}" \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/tok-%A_%a.out" \
  --error="${RUN_DIR}/logs/tok-%A_%a.err" \
  --export=ALL,RUN_DIR \
  "${RUN_DIR}/scripts/tokenize_topup.sbatch")
echo "tokenize_job_id=${TOK_JOB}"
echo "${TOK_JOB}" > "${RUN_DIR}/tokenize_job_id.txt"

UP_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal --cpus-per-task=8 --mem=32G --time=12:00:00 \
  --dependency="afterok:${TOK_JOB}" \
  --job-name=topup-up \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/upload-%j.out" \
  --error="${RUN_DIR}/logs/upload-%j.err" \
  --wrap="set -Eeuo pipefail; unset PREFIX || true; source ${RUN_DIR}/env.sh; export EDULLM_ROOT=${EDULLM_ROOT} RUN_DIR=${RUN_DIR}; source ${RUN_DIR}/scripts/prepare_aws_session_light.sh; source \${AWS_SESSION_ENV}; source ${RUN_DIR}/venv/bin/activate; python ${RUN_DIR}/scripts/finalize_olmohq_topup_upload.py --run-dir ${RUN_DIR} --bucket ${BUCKET} --prefix \${OLMOHQ_PREFIX}")
echo "upload_job_id=${UP_JOB}"
echo "${UP_JOB}" > "${RUN_DIR}/upload_job_id.txt"

echo "=== mixlaw validation ==="
sed -i 's/\r$//' "$STAGING/experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh"
bash "$STAGING/experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh" | tee /scratch/users/nzhao2/agent-runs/mixlaw-validation-latest-submit.log

echo "=== squeue ==="
squeue --me -o '%.18i %.9P %.30j %.8T %.10M %.6D %R' | head -80
REMOTE
