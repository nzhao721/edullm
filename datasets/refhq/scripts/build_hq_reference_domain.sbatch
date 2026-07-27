#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=hq-build
#SBATCH --output=logs/hq-build-%A_%a.out
#SBATCH --error=logs/hq-build-%A_%a.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${PLAN:?}" "${DOMAIN_LIST:?}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export HF_HOME="${RUN_DIR}/hf-cache"
export TOKENIZERS_PARALLELISM=false
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "${HF_HOME}"

if [[ -f "${RUN_DIR}/.hf_token" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '\r\n' < "${RUN_DIR}/.hf_token")"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  if [[ -z "${HF_TOKEN}" ]]; then
    echo "ERROR: ${RUN_DIR}/.hf_token is empty" >&2
    exit 1
  fi
  echo "HF_TOKEN loaded (len=${#HF_TOKEN})"
else
  echo "ERROR: missing ${RUN_DIR}/.hf_token (required for gated HF datasets)" >&2
  exit 1
fi

: "${HQ_SCRIPTS:=${RUN_DIR}/datasets/refhq/scripts}"

# shellcheck disable=SC1091
source "${HQ_SCRIPTS}/lib.sh"
dolma_hq_export_pythonpath "${RUN_DIR}"

read -r -a DOMAINS <<< "${DOMAIN_LIST}"
DOMAIN="${DOMAINS[${SLURM_ARRAY_TASK_ID}]}"
echo "build domain=${DOMAIN} task=${SLURM_ARRAY_TASK_ID} cpus=${SLURM_CPUS_PER_TASK}"

python "${HQ_SCRIPTS}/build_hq_reference_domain.py" \
  --plan "${PLAN}" \
  --domain "${DOMAIN}" \
  --processes "${SLURM_CPUS_PER_TASK}"
