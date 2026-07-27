#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=hq-finalize
#SBATCH --output=logs/hq-finalize-%j.out
#SBATCH --error=logs/hq-finalize-%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${PLAN:?}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

: "${HQ_SCRIPTS:=${RUN_DIR}/datasets/refhq/scripts}"

# shellcheck disable=SC1091
source "${RUN_DIR}/env.sh"
export RUN_DIR
if [[ -n "${AWS_SESSION_ENV:-}" && -f "${AWS_SESSION_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${AWS_SESSION_ENV}"
else
  export STAGING_ROOT="${STAGING_ROOT:-/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging}"
  # shellcheck disable=SC1091
  source "${RUN_DIR}/scripts/farmshare/prepare_aws_session_light.sh"
  # shellcheck disable=SC1090
  source "${AWS_SESSION_ENV}"
fi

# shellcheck disable=SC1091
source "${HQ_SCRIPTS}/lib.sh"
dolma_hq_export_pythonpath "${RUN_DIR}"

python "${HQ_SCRIPTS}/finalize_hq_reference_upload.py" \
  --plan "${PLAN}" \
  ${SKIP_UPLOAD:+--skip-upload}
