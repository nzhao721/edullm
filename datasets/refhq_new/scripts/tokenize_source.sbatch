#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=refhqn-tok
#SBATCH --output=logs/refhqn-tok-%A_%a.out
#SBATCH --error=logs/refhqn-tok-%A_%a.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${PLAN:?}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export HF_HOME="${RUN_DIR}/hf-cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}"

: "${REFHQ_NEW_SCRIPTS:=${RUN_DIR}/datasets/refhq_new/scripts}"
: "${SCRATCH_ROOT:=${RUN_DIR}}"
: "${TOKENIZE_TASKS:=${SCRATCH_ROOT}/manifests/tokenize_tasks.txt}"

# shellcheck disable=SC1091
source "${REFHQ_NEW_SCRIPTS}/lib.sh"
refhq_new_export_pythonpath "${RUN_DIR}"
refhq_new_load_hf_token "${RUN_DIR}"

TASK_INDEX="${SLURM_ARRAY_TASK_ID:?}"
echo "tokenize task_index=${TASK_INDEX} cpus=${SLURM_CPUS_PER_TASK} tasks=${TOKENIZE_TASKS}"

python "${REFHQ_NEW_SCRIPTS}/tokenize_source.py" \
  --plan "${PLAN}" \
  --task-index "${TASK_INDEX}" \
  --tasks-file "${TOKENIZE_TASKS}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --batch-size 64 \
  --tokenizer allenai/dolma2-tokenizer \
  --eos-token-id 100257
