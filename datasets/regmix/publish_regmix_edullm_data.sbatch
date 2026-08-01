#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=regmix-edullm-pub
#SBATCH --output=logs/regmix_edullm_publish_%j.out
#SBATCH --error=logs/regmix_edullm_publish_%j.er

# Constraints baked into publish_regmix_edullm_data.py:
#   - labels: tokens/<source>/… → entry.labels = {source} only (domain omitted)
#   - shards: max 1 GiB
#   - val: same fraction from every source (mix weights match full corpus)

set -Eeuo pipefail

: "${RUN_DIR:?}" "${REGMIX_ROOT:?}" "${STAGE_DIR:?}" "${EDULLM_ROOT:?}"

VENV="${VENV:-${RUN_DIR}/venv}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV:?}"

export PATH="${PATH}:${HOME}/.local/bin:${HOME}/tools/aws/bin"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

SCRIPTS="${SCRIPTS:-${EDULLM_ROOT}/datasets/regmix}"
TOKENIZED_ROOT="${TOKENIZED_ROOT:-${REGMIX_ROOT}/tokenized}"
TEXT_RUN_DIR="${TEXT_RUN_DIR:-${REGMIX_ROOT}}"
EDULLM_DATA_DIR="${EDULLM_DATA_DIR:-${RUN_DIR}/edullm-data}"

if [[ "${SKIP_EDULLM_DATA_INSTALL:-0}" != 1 ]]; then
  rm -rf "${EDULLM_DATA_DIR}"
  git clone --depth 1 --branch main https://github.com/edu-llm/edullm-data.git "${EDULLM_DATA_DIR}"
  pip install -q -e "${EDULLM_DATA_DIR}"
fi

python "${SCRIPTS}/publish_regmix_edullm_data.py" \
  --tokenized-root "${TOKENIZED_ROOT}" \
  --text-run-dir "${TEXT_RUN_DIR}" \
  --stage-dir "${STAGE_DIR}" \
  --shard-bytes 1073741824 \
  --val-fraction 0.0015 \
  --hash-workers 16 \
  --copy-workers 16 \
  --force
