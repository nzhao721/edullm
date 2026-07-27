#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=pool-tok-up
#SBATCH --output=logs/pool-tok-up-%j.out
#SBATCH --error=logs/pool-tok-up-%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${SRC_BUCKET:?}" "${SRC_PREFIX:?}"

export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"

SHARED_AWS_SESSION="/scratch/users/nzhao2/agent-runs/_shared/aws-session-dolma2-tok.env"
if [[ -f "${SHARED_AWS_SESSION}" ]]; then
  # shellcheck disable=SC1090
  source "${SHARED_AWS_SESSION}"
elif [[ -n "${AWS_SESSION_ENV:-}" && -f "${AWS_SESSION_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${AWS_SESSION_ENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
# shellcheck disable=SC1091
source "${RUN_DIR}/env.sh"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV:?}"

python "${RUN_DIR}/scripts/finalize_pool_tokenized_upload.py" \
  --run-dir "${RUN_DIR}" \
  --bucket "${SRC_BUCKET}" \
  --prefix "${SRC_PREFIX}"
