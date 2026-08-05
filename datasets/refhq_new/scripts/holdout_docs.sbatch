#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=refhqn-hold
#SBATCH --output=logs/refhqn-holdout-%j.out
#SBATCH --error=logs/refhqn-holdout-%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${PLAN:?}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

: "${REFHQ_NEW_SCRIPTS:=${RUN_DIR}/datasets/refhq_new/scripts}"

# shellcheck disable=SC1091
source "${REFHQ_NEW_SCRIPTS}/lib.sh"
refhq_new_export_pythonpath "${RUN_DIR}"

echo "holdout plan=${PLAN} fraction=0.0015 (0.15% docs per source/domain, seed from plan)"

python "${REFHQ_NEW_SCRIPTS}/holdout_docs.py" \
  --plan "${PLAN}"
