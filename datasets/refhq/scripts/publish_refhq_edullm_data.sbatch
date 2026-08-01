#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=refhq-edullm-pub
#SBATCH --output=logs/refhq_edullm_publish_%j.out
#SBATCH --error=logs/refhq_edullm_publish_%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${REFHQ_ROOT:?}" "${STAGE_DIR:?}" "${EDULLM_ROOT:?}"

VENV="${VENV:-${RUN_DIR}/venv}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV:?}"

export PATH="${PATH}:${HOME}/.local/bin:${HOME}/tools/aws/bin"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

HQ_SCRIPTS="${HQ_SCRIPTS:-${EDULLM_ROOT}/datasets/refhq/scripts}"
TOKENIZED_ROOT="${TOKENIZED_ROOT:-${REFHQ_ROOT}/tokenized}"
TEXT_RUN_DIR="${TEXT_RUN_DIR:-${REFHQ_ROOT}}"
MANIFEST="${MANIFEST:-${REFHQ_ROOT}/manifests/tokenized_manifest.json}"
EDULLM_DATA_DIR="${EDULLM_DATA_DIR:-${RUN_DIR}/edullm-data}"

rm -rf "${EDULLM_DATA_DIR}"
git clone --depth 1 --branch main https://github.com/edu-llm/edullm-data.git "${EDULLM_DATA_DIR}"
pip install -q -e "${EDULLM_DATA_DIR}"

STAGE_ARGS=(--force)

python "${HQ_SCRIPTS}/publish_refhq_edullm_data.py" \
  --tokenized-root "${TOKENIZED_ROOT}" \
  --text-run-dir "${TEXT_RUN_DIR}" \
  --manifest "${MANIFEST}" \
  --stage-dir "${STAGE_DIR}" \
  --hash-workers 16 \
  --copy-workers 16 \
  "${STAGE_ARGS[@]}"
