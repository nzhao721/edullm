#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Hashing and S3 multipart uploads share the eight worker slots.
#SBATCH --cpus-per-task=8
# Payload is streamed; memory is for concurrent hash/upload buffers only.
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=regmix-doc-publish
#SBATCH --output=logs/regmix_doc_publish_%j.out
#SBATCH --error=logs/regmix_doc_publish_%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${SCRIPTS:?}" "${TOKENIZED_ROOT:?}" "${STAGE_DIR:?}"
: "${DATASET_ID:?}" "${SELECTION_MANIFEST:?}" "${SELECTION_DESCRIPTION:?}" "${PURPOSE:?}"

# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV:?}"

export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

EDULLM_DATA_DIR="${EDULLM_DATA_DIR:-${RUN_DIR}/edullm-data}"
rm -rf "${EDULLM_DATA_DIR}"
git clone --depth 1 --branch main https://github.com/edu-llm/edullm-data.git "${EDULLM_DATA_DIR}"
pip install -q -e "${EDULLM_DATA_DIR}"

python "${SCRIPTS}/publish_regmix_doc_subset_edullm_data.py" \
  --tokenized-root "${TOKENIZED_ROOT}" \
  --stage-dir "${STAGE_DIR}" \
  --dataset-id "${DATASET_ID}" \
  --selection-manifest "${SELECTION_MANIFEST}" \
  --selection-description "${SELECTION_DESCRIPTION}" \
  --purpose "${PURPOSE}" \
  --hash-workers "${SLURM_CPUS_PER_TASK}" \
  --copy-workers "${SLURM_CPUS_PER_TASK}" \
  --force
