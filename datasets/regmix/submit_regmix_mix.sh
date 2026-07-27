#!/usr/bin/env bash
# Build RegMix-aligned 10B mix on FarmShare from edullm-dataset-olmohq, upload to edullm-dataset-regmix.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-regmix-10b-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
EDULLM_ROOT="${EDULLM_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
SRC_BUCKET="${SRC_BUCKET:-edullm-dataset-olmohq}"
SRC_PREFIX="${SRC_PREFIX:-olmo-mix-1124-30b}"
DST_BUCKET="${DST_BUCKET:-edullm-dataset-regmix}"
DST_PREFIX="${DST_PREFIX:-regmix-10b}"
SEED="${SEED:-42}"
# Prefer the local olmohq upsample mirror when present (hardlink/copy by size).
LOCAL_MIRROR="${LOCAL_MIRROR:-/scratch/users/${SUNET}/agent-runs/olmo-mix-upsample-20260723-103547/data}"
DOMAIN_LIST="${DOMAIN_LIST:-dclm arxiv starcoder pes2o open-web-math algebraic-stack wiki}"
DL_CONCURRENCY="${DL_CONCURRENCY:-80}"

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${RUN_DIR}/data" "${RUN_DIR}/plan" "${RUN_DIR}/trim"
cd "${RUN_DIR}"

# Sync pipeline scripts into the isolated run dir.
REGMIX_ROOT="${EDULLM_ROOT}/datasets/regmix"
DATASETS_SHARED="${EDULLM_ROOT}/datasets"
FARMSHARE="${EDULLM_ROOT}/scripts/farmshare"
cp -a "${REGMIX_ROOT}/plan_regmix_mix.py" "${RUN_DIR}/scripts/"
cp -a "${REGMIX_ROOT}/finalize_regmix_upload.py" "${RUN_DIR}/scripts/"
cp -a "${REGMIX_ROOT}/trim_regmix_domain.sbatch" "${RUN_DIR}/scripts/"
cp -a "${DATASETS_SHARED}/download_s3_shard.py" "${RUN_DIR}/scripts/"
cp -a "${DATASETS_SHARED}/download_s3_shard.sbatch" "${RUN_DIR}/scripts/"
cp -a "${DATASETS_SHARED}/trim_and_tokenize_regmix.py" "${RUN_DIR}/scripts/"
cp -a "${DATASETS_SHARED}/olmo_shard_utils.py" "${RUN_DIR}/scripts/"

# Venv
if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
fi
# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
pip install -U pip wheel
pip install boto3 tqdm transformers zstandard

# Mint AWS session env for compute nodes (broker profile on login node).
# Use the light helper — full prepare_aws_session.sh requires Dolma/HF bootstrap.
export EDULLM_ROOT RUN_DIR
# shellcheck disable=SC1091
source "${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV}"

# Optional: pull pool summary from S3 for accurate tokens/byte rates.
POOL_SUMMARY="${RUN_DIR}/plan/pool_summary.json"
aws s3 cp "s3://${SRC_BUCKET}/${SRC_PREFIX}/plan/summary.json" "${POOL_SUMMARY}" || true

python "${RUN_DIR}/scripts/plan_regmix_mix.py" \
  --src-bucket "${SRC_BUCKET}" \
  --src-prefix "${SRC_PREFIX}" \
  --seed "${SEED}" \
  --out-dir "${RUN_DIR}/plan" \
  --pool-summary "${POOL_SUMMARY}"

MANIFEST="${RUN_DIR}/plan/manifest.jsonl"
SUMMARY="${RUN_DIR}/plan/summary.json"
N=$(wc -l < "${MANIFEST}")
NDOMS=$(echo "${DOMAIN_LIST}" | wc -w)

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${RUN_DIR}/venv
MANIFEST=${MANIFEST}
SUMMARY=${SUMMARY}
LOCAL_ROOT=${RUN_DIR}/data
SRC_BUCKET=${SRC_BUCKET}
SRC_PREFIX=${SRC_PREFIX}
DST_BUCKET=${DST_BUCKET}
DST_PREFIX=${DST_PREFIX}
DOMAIN_LIST="${DOMAIN_LIST}"
LOCAL_MIRROR=${LOCAL_MIRROR}
EDULLM_ROOT=${EDULLM_ROOT}
AWS_SESSION_ENV=${AWS_SESSION_ENV}
N=${N}
EOF

echo "manifest_files=${N}"
echo "domains=${NDOMS}"
echo "RUN_DIR=${RUN_DIR}" | tee "${RUN_DIR}/RUN_DIR.txt"
cat "${SUMMARY}"

# Max-parallel download array.
DL_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --array=0-$((N - 1))%${DL_CONCURRENCY} \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR,VENV,MANIFEST,LOCAL_ROOT,SRC_BUCKET,LOCAL_MIRROR,AWS_SESSION_ENV \
  "${RUN_DIR}/scripts/download_s3_shard.sbatch")
echo "download_job_id=${DL_JOB}"
echo "${DL_JOB}" > "${RUN_DIR}/download_job_id.txt"

# Per-domain document trim after all downloads succeed.
TRIM_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --array=0-$((NDOMS - 1)) \
  --dependency="afterok:${DL_JOB}" \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR,VENV,MANIFEST,SUMMARY,DOMAIN_LIST \
  "${RUN_DIR}/scripts/trim_regmix_domain.sbatch")
echo "trim_job_id=${TRIM_JOB}"
echo "${TRIM_JOB}" > "${RUN_DIR}/trim_job_id.txt"

# Provision destination bucket + upload after all trims succeed.
# Refresh AWS session just before upload via wrap that re-sources prepare_aws_session.
FINAL_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task=8 \
  --mem=32G \
  --time=06:00:00 \
  --dependency="afterok:${TRIM_JOB}" \
  --job-name=regmix-up \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/upload-%j.out" \
  --error="${RUN_DIR}/logs/upload-%j.err" \
  --wrap="set -Eeuo pipefail; source ${RUN_DIR}/env.sh; export EDULLM_ROOT RUN_DIR; source ${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh; source ${AWS_SESSION_ENV}; source ${VENV}/bin/activate; python ${RUN_DIR}/scripts/finalize_regmix_upload.py --run-dir ${RUN_DIR} --dst-bucket ${DST_BUCKET} --dst-prefix ${DST_PREFIX}")
echo "upload_job_id=${FINAL_JOB}"
echo "${FINAL_JOB}" > "${RUN_DIR}/upload_job_id.txt"

echo "submitted download=${DL_JOB} trim=${TRIM_JOB} upload=${FINAL_JOB}"
