#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=regmix-trim
#SBATCH --output=logs/trim-%A_%a.out
#SBATCH --error=logs/trim-%A_%a.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${MANIFEST:?}" "${SUMMARY:?}" "${DOMAIN_LIST:?}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export HF_HOME="${RUN_DIR}/hf-cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${RUN_DIR}/tokenized"

if [[ -f "${RUN_DIR}/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '\r\n' < "${RUN_DIR}/.hf_token")"
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

read -r -a DOMAINS <<< "${DOMAIN_LIST}"
DOMAIN="${DOMAINS[${SLURM_ARRAY_TASK_ID}]}"
echo "trim+tokenize domain=${DOMAIN} task=${SLURM_ARRAY_TASK_ID} cpus=${SLURM_CPUS_PER_TASK}"
echo "tokenizer=allenai/dolma2-tokenizer eos=100257"

# Wipe partial artifacts from the cancelled wrong-tokenizer run.
rm -rf "${RUN_DIR}/trim/${DOMAIN}"
rm -rf "${RUN_DIR}/tokenized/${DOMAIN}"

python "${RUN_DIR}/scripts/trim_and_tokenize_regmix.py" \
  --run-dir "${RUN_DIR}" \
  --summary "${SUMMARY}" \
  --manifest "${MANIFEST}" \
  --domain "${DOMAIN}" \
  --seed 42 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --batch-size 64 \
  --tokenizer allenai/dolma2-tokenizer \
  --eos-token-id 100257 \
  --force
