#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=refhq-edullm-pub
#SBATCH --output=logs/refhq_edullm_publish_%j.out
#SBATCH --error=logs/refhq_edullm_publish_%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${REFHQ_ROOT:?}" "${STAGE_DIR:?}" "${EDULLM_ROOT:?}"

# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV:?}"

export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

HQ_SCRIPTS="${HQ_SCRIPTS:-${REFHQ_ROOT}/datasets/refhq/scripts}"
TOKENIZED_ROOT="${TOKENIZED_ROOT:-${REFHQ_ROOT}/tokenized}"
MANIFEST="${MANIFEST:-${REFHQ_ROOT}/manifests/tokenized_manifest.json}"
EDULLM_DATA_DIR="${EDULLM_DATA_DIR:-${RUN_DIR}/edullm-data}"

rm -rf "${EDULLM_DATA_DIR}"
git clone --depth 1 --branch main https://github.com/edu-llm/edullm-data.git "${EDULLM_DATA_DIR}"
pip install -q -e "${EDULLM_DATA_DIR}"

STAGE_ARGS=(--force)

python "${HQ_SCRIPTS}/publish_refhq_edullm_data.py" \
  --tokenized-root "${TOKENIZED_ROOT}" \
  --manifest "${MANIFEST}" \
  --stage-dir "${STAGE_DIR}" \
  --hash-workers 16 \
  --copy-workers 16 \
  "${STAGE_ARGS[@]}"
