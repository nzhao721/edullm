#!/usr/bin/env bash
# Build a peak-sized working pool for mixlaw 370M validation training.
#
# Fetches token shards from published+validated
#   s3://edullm-data/pretrain/olmo-127b  (default; resolve_latest + dataset_paths)
# and concatenates per-domain memmaps. Never reads s3://edullm-datasets/.
# Does not assume a pre-existing FarmShare / laptop pool or ladder venv.
#
# Bootstraps a CPU venv in RUN_DIR (edullm-data + boto3). No ladder GPU venv.
#
# Sole supported path: DomainMixtureStream at recipe weights
# (validation_mixtures_10b.json); do not use build_mixture_data.py slices.
#
# Usage (FarmShare login node with AWS session env available):
#   bash experiments/skill-dag/mixlaw/submit_mixlaw_validation_pool.sh
set -Eeuo pipefail

SUNET="${SUNET:-${USER:-nzhao2}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EDULLM_ROOT="${EDULLM_ROOT:-${REPO_ROOT}}"

RUN_NAME="${RUN_NAME:-mixlaw-validation-pool-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
DATASET_ID="${DATASET_ID:-pretrain/olmo-127b}"
BUDGET_TOKENS="${BUDGET_TOKENS:-10000000000}"
BUILD_WORKERS="${BUILD_WORKERS:-4}"
EDULLM_DATA_PKG="${EDULLM_DATA_PKG:-edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0}"

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${RUN_DIR}/pool" "${RUN_DIR}/plan"
cd "${RUN_DIR}"

for f in mixlaw_common.py stage_validation_pool_from_edullm_data.py validation_mixtures_10b.json; do
  cp -a "${SCRIPT_DIR}/${f}" "${RUN_DIR}/scripts/"
done
if [[ -f "${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" ]]; then
  cp -a "${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" "${RUN_DIR}/scripts/" 2>/dev/null || true
  cp -a "${EDULLM_ROOT}/scripts/farmshare/write_aws_session_env.py" "${RUN_DIR}/scripts/" 2>/dev/null || true
fi
sed -i 's/\r$//' "${RUN_DIR}/scripts/"*.{sh,py} 2>/dev/null || true

if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
fi
# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
pip install -U pip wheel
pip install boto3 tqdm numpy
# shellcheck disable=SC2086
pip install ${EDULLM_DATA_PKG}

export EDULLM_ROOT RUN_DIR
unset PREFIX || true
AWS_SESSION_ENV="${AWS_SESSION_ENV:-}"
if [[ -f "${RUN_DIR}/scripts/prepare_aws_session_light.sh" ]]; then
  # shellcheck disable=SC1091
  source "${RUN_DIR}/scripts/prepare_aws_session_light.sh"
  # shellcheck disable=SC1090
  source "${AWS_SESSION_ENV}"
fi

cp -a "${RUN_DIR}/scripts/validation_mixtures_10b.json" "${RUN_DIR}/plan/"

VERSION_ARG=""
if [[ -n "${DATASET_VERSION:-}" ]]; then
  VERSION_ARG="--dataset-version ${DATASET_VERSION}"
fi

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${RUN_DIR}/venv
AWS_SESSION_ENV=${AWS_SESSION_ENV:-}
DATASET_ID=${DATASET_ID}
DATASET_VERSION=${DATASET_VERSION:-}
BUDGET_TOKENS=${BUDGET_TOKENS}
EOF

POOL_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task="${BUILD_WORKERS}" \
  --mem=128G \
  --time=12:00:00 \
  --job-name=mixlaw-pool \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/pool-%j.out" \
  --error="${RUN_DIR}/logs/pool-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; export PATH=\"\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}\"; source ${RUN_DIR}/env.sh; [[ -n \"\${AWS_SESSION_ENV}\" && -f \${AWS_SESSION_ENV} ]] && source \${AWS_SESSION_ENV}; source \${VENV}/bin/activate; cd ${RUN_DIR}/scripts; python stage_validation_pool_from_edullm_data.py --out-dir ${RUN_DIR}/pool --mixtures-json ${RUN_DIR}/plan/validation_mixtures_10b.json --budget-tokens ${BUDGET_TOKENS} --dataset-id ${DATASET_ID} ${VERSION_ARG}'")

echo "pool_job_id=${POOL_JOB}"
echo "RUN_DIR=${RUN_DIR}"
echo "POOL_DIR=${RUN_DIR}/pool"
echo "DATASET_ID=${DATASET_ID}"
echo "${POOL_JOB}" > "${RUN_DIR}/pool_job_id.txt"
