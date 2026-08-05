#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=refhqn-eng
#SBATCH --output=logs/refhqn-english-%A_%a.out
#SBATCH --error=logs/refhqn-english-%A_%a.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${PLAN:?}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export HF_HOME="${RUN_DIR}/hf-cache"
mkdir -p "${HF_HOME}"

: "${REFHQ_NEW_SCRIPTS:=${RUN_DIR}/datasets/refhq_new/scripts}"
: "${SCRATCH_ROOT:=${RUN_DIR}}"
: "${ENGLISH_TASKS:=${SCRATCH_ROOT}/manifests/english_tasks.txt}"

# shellcheck disable=SC1091
source "${REFHQ_NEW_SCRIPTS}/lib.sh"
refhq_new_export_pythonpath "${RUN_DIR}"

TASK_INDEX="${SLURM_ARRAY_TASK_ID:?}"
echo "english-filter task_index=${TASK_INDEX} cpus=${SLURM_CPUS_PER_TASK} tasks=${ENGLISH_TASKS}"

# Avoid nltk/defusedxml cwd import block on FarmShare scratch (see process._run_dolma).
cd /tmp

python "${REFHQ_NEW_SCRIPTS}/dolma_english_filter.py" \
  --plan "${PLAN}" \
  --task-index "${TASK_INDEX}" \
  --tasks-file "${ENGLISH_TASKS}" \
  --processes "${SLURM_CPUS_PER_TASK}"
