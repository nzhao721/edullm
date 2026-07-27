#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=hq-tok
#SBATCH --output=logs/hq-tok-%A_%a.out
#SBATCH --error=logs/hq-tok-%A_%a.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${PLAN:?}" "${DOMAIN_LIST:?}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export HF_HOME="${RUN_DIR}/hf-cache"
export TOKENIZERS_PARALLELISM=false
: "${HQ_SCRIPTS:=${RUN_DIR}/datasets/refhq/scripts}"

# shellcheck disable=SC1091
source "${HQ_SCRIPTS}/lib.sh"
dolma_hq_export_pythonpath "${RUN_DIR}"

read -r -a DOMAINS <<< "${DOMAIN_LIST}"
DOMAIN="${DOMAINS[${SLURM_ARRAY_TASK_ID}]}"
echo "tokenize domain=${DOMAIN} task=${SLURM_ARRAY_TASK_ID} cpus=${SLURM_CPUS_PER_TASK}"
echo "tokenizer=allenai/dolma2-tokenizer eos=100257"

python "${HQ_SCRIPTS}/tokenize_hq_reference_domain.py" \
  --plan "${PLAN}" \
  --domain "${DOMAIN}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --batch-size 64 \
  --tokenizer allenai/dolma2-tokenizer \
  --eos-token-id 100257
