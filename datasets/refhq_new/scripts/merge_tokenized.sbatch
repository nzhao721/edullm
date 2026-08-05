#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=refhqn-merge
#SBATCH --output=logs/refhqn-merge-%j.out
#SBATCH --error=logs/refhqn-merge-%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${PLAN:?}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
: "${REFHQ_NEW_SCRIPTS:=${RUN_DIR}/datasets/refhq_new/scripts}"
# shellcheck disable=SC1091
source "${REFHQ_NEW_SCRIPTS}/lib.sh"
refhq_new_export_pythonpath "${RUN_DIR}"

python "${REFHQ_NEW_SCRIPTS}/merge_tokenized.py" --plan "${PLAN}"
