#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=hq-dl
#SBATCH --output=logs/hq-download-%A_%a.out
#SBATCH --error=logs/hq-download-%A_%a.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${PLAN:?}" "${DOMAIN_LIST:?}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export HF_HOME="${RUN_DIR}/hf-cache"
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "${HF_HOME}"

if [[ -f "${RUN_DIR}/.hf_token" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '\r\n' < "${RUN_DIR}/.hf_token")"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

: "${HQ_SCRIPTS:=${RUN_DIR}/datasets/refhq/scripts}"

# shellcheck disable=SC1091
source "${HQ_SCRIPTS}/lib.sh"
dolma_hq_export_pythonpath "${RUN_DIR}"

read -r -a DOMAINS <<< "${DOMAIN_LIST}"
DOMAIN="${DOMAINS[${SLURM_ARRAY_TASK_ID}]}"
echo "download domain=${DOMAIN} task=${SLURM_ARRAY_TASK_ID}"

python "${HQ_SCRIPTS}/download_hq_source.py" \
  --plan "${PLAN}" \
  --domain "${DOMAIN}"
