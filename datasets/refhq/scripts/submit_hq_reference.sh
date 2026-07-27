#!/usr/bin/env bash
# Orchestrate HQ reference corpus on FarmShare scratch (CPU only; no laptop processing).
# Max parallelization: one Slurm array task per domain with no % concurrency throttle.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-hq-reference-v1}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/users/${SUNET}/${RUN_NAME}}"
RUN_DIR="${RUN_DIR:-${SCRATCH_ROOT}}"
STAGING_ROOT="${STAGING_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
DOMAIN_LIST="${DOMAIN_LIST:-dclm starcoder pes2o arxiv open-web-math algebraic-stack wiki}"
SEED="${SEED:-42}"
HF_TOKEN_SRC="${HF_TOKEN_SRC:-/scratch/users/${SUNET}/agent-runs/olmo-mix-30b-20260722/.hf_token}"
HQ_SCRIPTS="${RUN_DIR}/datasets/refhq/scripts"

# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/manifests"
cd "${RUN_DIR}"

dolma_hq_sync_to_run "${STAGING_ROOT}" "${RUN_DIR}"
dolma_hq_export_pythonpath "${RUN_DIR}"

if [[ -f "${HF_TOKEN_SRC}" && ! -f "${RUN_DIR}/.hf_token" ]]; then
  cp -a "${HF_TOKEN_SRC}" "${RUN_DIR}/.hf_token"
  chmod 600 "${RUN_DIR}/.hf_token"
  echo "copied HF token from ${HF_TOKEN_SRC}"
fi

if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
fi
# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
pip install -U pip wheel
pip install \
  "huggingface_hub[hf_transfer]" hf_transfer \
  "datasets>=2.19,<3" "tokenizers>=0.21.0" "transformers>=4.49.0" tqdm zstandard numpy \
  "dolma[code]==1.1.2" pyyaml boto3 awscli || true
pip install -U 'datasets>=2.19,<3' 'huggingface_hub>=0.23'

python "${HQ_SCRIPTS}/plan_hq_reference.py" \
  --scratch-root "${SCRATCH_ROOT}" \
  --seed "${SEED}"

PLAN="${SCRATCH_ROOT}/manifests/plan.json"
read -r -a DOMAINS <<< "${DOMAIN_LIST}"
N=${#DOMAINS[@]}

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${RUN_DIR}/venv
PLAN=${PLAN}
DOMAIN_LIST="${DOMAIN_LIST}"
SCRATCH_ROOT=${SCRATCH_ROOT}
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

echo "submitted HQ reference pipeline under ${SCRATCH_ROOT}"
echo "parallel_domains=${N} download=${DL_JOB} build=${BUILD_JOB} finalize=${FIN_JOB}"
