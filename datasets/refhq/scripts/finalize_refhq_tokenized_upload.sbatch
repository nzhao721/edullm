#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=hq-tok-up
#SBATCH --output=logs/hq-tok-up-%j.out
#SBATCH --error=logs/hq-tok-up-%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${PLAN:?}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
# shellcheck disable=SC1091
source "${RUN_DIR}/env.sh"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV:?}"

: "${HQ_SCRIPTS:=${RUN_DIR}/datasets/refhq/scripts}"

# shellcheck disable=SC1091
source "${HQ_SCRIPTS}/lib.sh"
dolma_hq_export_pythonpath "${RUN_DIR}"

python "${HQ_SCRIPTS}/finalize_refhq_tokenized_upload.py" \
  --plan "${PLAN}"
