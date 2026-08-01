#!/usr/bin/env bash
# Build a peak-sized working pool for the 24 DataDecide-60M mixing-law probes
# from published+validated s3://edullm-data (default: pretrain/olmo-127b).
#
# Sole supported path: DomainMixtureStream at recipe weights from mixtures.json
# (peak pool only; do not materialize per-mix slices via build_mixture_data.py).
# Does not read s3://edullm-datasets/.
#
# Ephemeral scratch: RUN_DIR may start empty and be wiped after the job.
# Stage into RUN_DIR/pool; create a job-local venv (no assumed FarmShare venv).
#
# Required:
#   RUN_DIR   ephemeral job directory (scratch OK)
#
# Optional:
#   EDULLM_ROOT       repo root (default: three levels above this script)
#   DATASET_ID        default pretrain/olmo-127b
#   DATASET_VERSION   pin; default resolve_latest
#   TOKENS_PER_PARAM  default 5
#
# Usage (FarmShare login node):
#   RUN_DIR=/scratch/users/$USER/agent-runs/mixlaw-pilot-pool-$(date +%Y%m%d-%H%M%S) \
#     bash experiments/skill-dag/mixlaw/submit_mixlaw_pilot_pool.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

: "${RUN_DIR:?Set RUN_DIR to an ephemeral job directory (scratch OK; wiped after)}"

EDULLM_ROOT="${EDULLM_ROOT:-${REPO_ROOT}}"
DATASET_ID="${DATASET_ID:-pretrain/olmo-127b}"
DATASET_VERSION="${DATASET_VERSION:-}"
TOKENS_PER_PARAM="${TOKENS_PER_PARAM:-5}"
BUILD_WORKERS="${BUILD_WORKERS:-4}"
EDULLM_DATA_PKG="${EDULLM_DATA_PKG:-edullm-data @ git+https://github.com/edu-llm/edullm-data@main}"

MIXLAW_ROOT="${SCRIPT_DIR}"

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${RUN_DIR}/pool" "${RUN_DIR}/plan"
cd "${RUN_DIR}"

for f in mixlaw_common.py stage_working_pool_from_edullm_data.py mixtures.json; do
  cp -a "${MIXLAW_ROOT}/${f}" "${RUN_DIR}/scripts/"
done
if [[ -f "${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" ]]; then
  cp -a "${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" "${RUN_DIR}/scripts/"
  cp -a "${EDULLM_ROOT}/scripts/farmshare/write_aws_session_env.py" "${RUN_DIR}/scripts/"
fi
sed -i 's/\r$//' "${RUN_DIR}/scripts/"*.{sh,py} 2>/dev/null || true

# Job-local venv — never assume a persistent FarmShare/ladder venv.
POOL_VENV="${POOL_VENV:-${RUN_DIR}/venv}"
if [[ ! -x "${POOL_VENV}/bin/python" ]]; then
  python3 -m venv "${POOL_VENV}"
fi
# shellcheck disable=SC1091
source "${POOL_VENV}/bin/activate"
pip install -U pip wheel
pip install boto3 tqdm numpy
# shellcheck disable=SC2086
pip install ${EDULLM_DATA_PKG}

export EDULLM_ROOT RUN_DIR
unset PREFIX || true
AWS_SESSION_ENV=""
if [[ -f "${RUN_DIR}/scripts/prepare_aws_session_light.sh" ]]; then
  # shellcheck disable=SC1091
  source "${RUN_DIR}/scripts/prepare_aws_session_light.sh"
  # shellcheck disable=SC1090
  [[ -n "${AWS_SESSION_ENV:-}" && -f "${AWS_SESSION_ENV}" ]] && source "${AWS_SESSION_ENV}"
fi

cp -a "${RUN_DIR}/scripts/mixtures.json" "${RUN_DIR}/plan/"

BUDGET_TOKENS="$("${POOL_VENV}/bin/python" -c "import sys; sys.path.insert(0, '${RUN_DIR}/scripts'); from mixlaw_common import token_budget; print(token_budget(${TOKENS_PER_PARAM})[2])")"

VERSION_ARG=""
if [[ -n "${DATASET_VERSION}" ]]; then
  VERSION_ARG="--dataset-version ${DATASET_VERSION}"
fi

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${POOL_VENV}
AWS_SESSION_ENV=${AWS_SESSION_ENV:-}
DATASET_ID=${DATASET_ID}
DATASET_VERSION=${DATASET_VERSION}
BUDGET_TOKENS=${BUDGET_TOKENS}
TOKENS_PER_PARAM=${TOKENS_PER_PARAM}
EOF

AWS_BOOT=""
if [[ -n "${AWS_SESSION_ENV:-}" ]]; then
  AWS_BOOT="[[ -f \${AWS_SESSION_ENV} ]] && source \${AWS_SESSION_ENV};"
fi

POOL_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task="${BUILD_WORKERS}" \
  --mem=128G \
  --time=12:00:00 \
  --job-name=mixlaw-pilot-pool \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/pool-%j.out" \
  --error="${RUN_DIR}/logs/pool-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; export PATH=\"\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}\"; source ${RUN_DIR}/env.sh; ${AWS_BOOT} source \${VENV}/bin/activate; cd ${RUN_DIR}/scripts; python stage_working_pool_from_edullm_data.py --dataset-id ${DATASET_ID} ${VERSION_ARG} --out-dir ${RUN_DIR}/pool --mixtures-json ${RUN_DIR}/plan/mixtures.json --budget-tokens ${BUDGET_TOKENS}'")

echo "pool_job_id=${POOL_JOB}"
echo "RUN_DIR=${RUN_DIR}"
echo "POOL_DIR=${RUN_DIR}/pool"
echo "dataset_id=${DATASET_ID}"
echo "budget_tokens=${BUDGET_TOKENS}"
echo "${POOL_JOB}" > "${RUN_DIR}/pool_job_id.txt"
