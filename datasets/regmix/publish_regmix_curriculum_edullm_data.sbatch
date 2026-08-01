#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Four ~20 MiB rank arrays plus their identity validation permutations: 2 CPUs,
# 4 GiB, and 30 minutes cover stage, manifest hashing, and landing upload.
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=regmix-curr-pub
#SBATCH --output=logs/regmix_curriculum_edullm_publish_%j.out
#SBATCH --error=logs/regmix_curriculum_edullm_publish_%j.er

# Publishes four token-order/v1 groups (compression, flesch, mtld, learnability)
# as curriculum/regmix-370m. Requires pretrain/regmix-10b on edullm-data and
# ranked_chunks_*.npy from build_curriculum_index.py under INDEX_DIR.

set -Eeuo pipefail

: "${RUN_DIR:?}" "${INDEX_DIR:?}" "${STAGE_DIR:?}" "${EDULLM_ROOT:?}"

# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV:?}"

export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

SCRIPTS="${SCRIPTS:-${RUN_DIR}/scripts/regmix}"
EDULLM_DATA_DIR="${EDULLM_DATA_DIR:-${RUN_DIR}/edullm-data}"
DRY_RUN="${DRY_RUN:-0}"
PARENT_VERSION="${PARENT_VERSION:-}"

for metric in compression_ratio flesch mtld learnability; do
  if [[ ! -f "${INDEX_DIR}/ranked_chunks_${metric}.npy" ]]; then
    echo "ERROR: missing ${INDEX_DIR}/ranked_chunks_${metric}.npy" >&2
    echo "Run build_curriculum_index.py first (see experiments/curriculum/)." >&2
    exit 1
  fi
done
if [[ ! -f "${INDEX_DIR}/curriculum_manifest.json" ]]; then
  echo "ERROR: missing ${INDEX_DIR}/curriculum_manifest.json parent identity" >&2
  exit 1
fi

rm -rf "${EDULLM_DATA_DIR}"
git clone --depth 1 --branch main https://github.com/edu-llm/edullm-data.git "${EDULLM_DATA_DIR}"
pip install -q -e "${EDULLM_DATA_DIR}"

PUBLISH_ARGS=(
  --index-dir "${INDEX_DIR}"
  --stage-dir "${STAGE_DIR}"
  --force
)
if [[ "${DRY_RUN}" == "1" ]]; then
  PUBLISH_ARGS+=(--dry-run)
fi
if [[ -n "${PARENT_VERSION}" ]]; then
  PUBLISH_ARGS+=(--parent-version "${PARENT_VERSION}")
fi

python "${SCRIPTS}/publish_regmix_curriculum_edullm_data.py" "${PUBLISH_ARGS[@]}"
