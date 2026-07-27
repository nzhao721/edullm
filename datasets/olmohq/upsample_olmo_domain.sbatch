#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=olmo-up
#SBATCH --output=logs/upsample-%A_%a.out
#SBATCH --error=logs/upsample-%A_%a.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${MANIFEST:?}" "${SUMMARY:?}"
: "${DOMAIN_LIST:?}"

source "${VENV}/bin/activate"
export HF_HOME="${RUN_DIR}/hf-cache"
export TOKENIZERS_PARALLELISM=false
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "${HF_HOME}"

if [[ -f "${RUN_DIR}/.hf_token" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '\r\n' < "${RUN_DIR}/.hf_token")"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

read -r -a DOMAINS <<< "${DOMAIN_LIST}"
DOMAIN="${DOMAINS[${SLURM_ARRAY_TASK_ID}]}"
echo "upsampling domain=${DOMAIN} task=${SLURM_ARRAY_TASK_ID} cpus=${SLURM_CPUS_PER_TASK}"

python "${RUN_DIR}/scripts/upsample_olmo_domain.py" \
  --run-dir "${RUN_DIR}" \
  --summary "${SUMMARY}" \
  --manifest "${MANIFEST}" \
  --domain "${DOMAIN}" \
  --seed 42 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --batch-size 64 \
  --tokenizer allenai/OLMo-2-0425-1B \
  --force
