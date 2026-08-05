#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=refhqn-dl
#SBATCH --output=logs/refhqn-download-%A_%a.out
#SBATCH --error=logs/refhqn-download-%A_%a.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${PLAN:?}" "${SOURCE_LIST:?}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export HF_HOME="${RUN_DIR}/hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "${HF_HOME}"

: "${REFHQ_NEW_SCRIPTS:=${RUN_DIR}/datasets/refhq_new/scripts}"

# shellcheck disable=SC1091
source "${REFHQ_NEW_SCRIPTS}/lib.sh"
refhq_new_export_pythonpath "${RUN_DIR}"
refhq_new_load_hf_token "${RUN_DIR}"

read -r -a SOURCES <<< "${SOURCE_LIST}"
SOURCE="${SOURCES[${SLURM_ARRAY_TASK_ID}]}"
echo "download source=${SOURCE} task=${SLURM_ARRAY_TASK_ID}"

python "${REFHQ_NEW_SCRIPTS}/download_hf_source.py" \
  --plan "${PLAN}" \
  --source "${SOURCE}"
