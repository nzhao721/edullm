#!/usr/bin/env bash
# RegMix-weighted ~5.514B HQ reference on FarmShare -> s3://edullm-datasets/refhq/
# Max parallelization: all 7 domains download+build concurrently (no % throttle).
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-refhq-regmix-5p5b-v1}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/users/${SUNET}/${RUN_NAME}}"
RUN_DIR="${RUN_DIR:-${SCRATCH_ROOT}}"
STAGING_ROOT="${STAGING_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
REUSE_ROOT="${REUSE_ROOT:-/scratch/users/${SUNET}/hq-reference-v1}"
DOMAIN_LIST="${DOMAIN_LIST:-dclm starcoder pes2o arxiv open-web-math algebraic-stack wiki}"
SEED="${SEED:-42}"
BUDGET_PROFILE="${BUDGET_PROFILE:-regmix-5p5}"
S3_BUCKET="${S3_BUCKET:-edullm-datasets}"
S3_PREFIX="${S3_PREFIX:-refhq/refhq-regmix-5p5b-v1}"
HF_TOKEN_SRC="${HF_TOKEN_SRC:-${REUSE_ROOT}/.hf_token}"
HQ_SCRIPTS="${RUN_DIR}/datasets/refhq/scripts"

# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/manifests" "${RUN_DIR}/raw"
cd "${RUN_DIR}"

if [[ -d "${REUSE_ROOT}/venv" && ! -e "${RUN_DIR}/venv" ]]; then
  ln -s "${REUSE_ROOT}/venv" "${RUN_DIR}/venv"
  echo "linked venv from ${REUSE_ROOT}"
fi
if [[ -d "${REUSE_ROOT}/hf-cache" && ! -e "${RUN_DIR}/hf-cache" ]]; then
  ln -s "${REUSE_ROOT}/hf-cache" "${RUN_DIR}/hf-cache"
  echo "linked hf-cache from ${REUSE_ROOT}"
fi
if [[ -f "${HF_TOKEN_SRC}" && ! -f "${RUN_DIR}/.hf_token" ]]; then
  cp -a "${HF_TOKEN_SRC}" "${RUN_DIR}/.hf_token"
  chmod 600 "${RUN_DIR}/.hf_token"
  echo "copied HF token from ${HF_TOKEN_SRC}"
fi

for domain in starcoder pes2o arxiv open-web-math algebraic-stack wiki; do
  if [[ -d "${REUSE_ROOT}/raw/${domain}" && ! -e "${RUN_DIR}/raw/${domain}" ]]; then
    ln -s "${REUSE_ROOT}/raw/${domain}" "${RUN_DIR}/raw/${domain}"
    echo "linked raw/${domain} from ${REUSE_ROOT}"
  fi
done
mkdir -p "${RUN_DIR}/raw/dclm"

dolma_hq_sync_to_run "${STAGING_ROOT}" "${RUN_DIR}"
dolma_hq_export_pythonpath "${RUN_DIR}"

if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
  # shellcheck disable=SC1091
  source "${RUN_DIR}/venv/bin/activate"
  pip install -U pip wheel
  pip install \
    "huggingface_hub[hf_transfer]" hf_transfer \
    "datasets>=2.19,<3" "tokenizers>=0.21.0" "transformers>=4.49.0" tqdm zstandard numpy \
    "dolma[code]==1.1.2" pyyaml boto3 awscli || true
  pip install -U 'datasets>=2.19,<3' 'huggingface_hub>=0.23'
else
  # shellcheck disable=SC1091
  source "${RUN_DIR}/venv/bin/activate"
fi

python "${HQ_SCRIPTS}/plan_hq_reference.py" \
  --scratch-root "${SCRATCH_ROOT}" \
  --seed "${SEED}" \
  --budget-profile "${BUDGET_PROFILE}" \
  --s3-bucket "${S3_BUCKET}" \
  --s3-prefix "${S3_PREFIX}"

PLAN="${SCRATCH_ROOT}/manifests/plan.json"
read -r -a DOMAINS <<< "${DOMAIN_LIST}"
N=${#DOMAINS[@]}

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${RUN_DIR}/venv
PLAN=${PLAN}
DOMAIN_LIST="${DOMAIN_LIST}"
SCRATCH_ROOT=${SCRATCH_ROOT}
STAGING_ROOT=${STAGING_ROOT}
S3_BUCKET=${S3_BUCKET}
S3_PREFIX=${S3_PREFIX}
HQ_SCRIPTS=${HQ_SCRIPTS}
EOF

# shellcheck disable=SC1091
source "${RUN_DIR}/env.sh"
dolma_hq_export_pythonpath "${RUN_DIR}"

python "${HQ_SCRIPTS}/smoke_code_copyright_strip.py" --fixture

DL_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --array=0-$((N - 1)) \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",VENV="${VENV}",PLAN="${PLAN}",DOMAIN_LIST="${DOMAIN_LIST}",HQ_SCRIPTS="${HQ_SCRIPTS}" \
  "${HQ_SCRIPTS}/download_hq_source.sbatch")
echo "download_job=${DL_JOB}"

BUILD_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterok:${DL_JOB} \
  --array=0-$((N - 1)) \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",VENV="${VENV}",PLAN="${PLAN}",DOMAIN_LIST="${DOMAIN_LIST}",HQ_SCRIPTS="${HQ_SCRIPTS}" \
  "${HQ_SCRIPTS}/build_hq_reference_domain.sbatch")
echo "build_job=${BUILD_JOB}"

FIN_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterok:${BUILD_JOB} \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",VENV="${VENV}",PLAN="${PLAN}",HQ_SCRIPTS="${HQ_SCRIPTS}" \
  "${HQ_SCRIPTS}/finalize_hq_reference_upload.sbatch")
echo "finalize_job=${FIN_JOB}"

echo "submitted RegMix 5.5B HQ reference under ${SCRATCH_ROOT}"
echo "s3://${S3_BUCKET}/${S3_PREFIX}/"
echo "parallel_domains=${N} download=${DL_JOB} build=${BUILD_JOB} finalize=${FIN_JOB}"
