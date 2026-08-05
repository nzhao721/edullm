#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=refhqn-norm
#SBATCH --output=logs/refhqn-normalize-%A_%a.out
#SBATCH --error=logs/refhqn-normalize-%A_%a.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${PLAN:?}" "${SOURCE_LIST:?}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export HF_HOME="${RUN_DIR}/hf-cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}"

: "${REFHQ_NEW_SCRIPTS:=${RUN_DIR}/datasets/refhq_new/scripts}"

# shellcheck disable=SC1091
source "${REFHQ_NEW_SCRIPTS}/lib.sh"
refhq_new_export_pythonpath "${RUN_DIR}"
refhq_new_load_hf_token "${RUN_DIR}"

read -r -a SOURCES <<< "${SOURCE_LIST}"
SOURCE="${SOURCES[${SLURM_ARRAY_TASK_ID}]}"
echo "normalize source=${SOURCE} task=${SLURM_ARRAY_TASK_ID}"

python "${REFHQ_NEW_SCRIPTS}/normalize_filter_source.py" \
  --plan "${PLAN}" \
  --source "${SOURCE}"
