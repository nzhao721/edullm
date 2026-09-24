#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=refhq-new-fin
#SBATCH --output=logs/refhq_new_finalize_%j.out
#SBATCH --error=logs/refhq_new_finalize_%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${PLAN:?}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
# shellcheck disable=SC1091
source "${RUN_DIR}/env.sh" 2>/dev/null || true

: "${REFHQ_NEW_SCRIPTS:=${RUN_DIR}/datasets/refhq_new/scripts}"
STAGE_DIR="${STAGE_DIR:-${RUN_DIR}/publish-stage}"

export PATH="${PATH}:${HOME}/.local/bin:${HOME}/tools/aws/bin"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

# shellcheck disable=SC1091
source "${REFHQ_NEW_SCRIPTS}/lib.sh"
refhq_new_export_pythonpath "${RUN_DIR}"

python "${REFHQ_NEW_SCRIPTS}/finalize_upload.py" \
  --plan "${PLAN}" \
  --stage-dir "${STAGE_DIR}" \
  --force
