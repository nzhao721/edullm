#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# The indexer is single-process and does not tokenize parent bytes.
#SBATCH --cpus-per-task=2
# Metric joins and parent chunk metadata are held in memory.
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=regmix-curr-index
#SBATCH --output=logs/regmix_curriculum_index_%j.out
#SBATCH --error=logs/regmix_curriculum_index_%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${REGMIX_ROOT:?}" "${INDEX_DIR:?}" "${SCRIPTS:?}"
: "${PARENT_LAYOUT:?captured dataset_paths parent layout is required}"
: "${PARENT_VERSION:?pinned pretrain/regmix-10b version is required}"
: "${PARENT_MANIFEST_SHA256:?pinned parent manifest hash is required}"

# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"

LABELS_ROOT="${LABELS_ROOT:-${REGMIX_ROOT}/labels}"
LM_LABELS_ROOT="${LM_LABELS_ROOT:-${REGMIX_ROOT}/lm_labels/labels}"

for path in \
  "${LABELS_ROOT}/metrics_index.jsonl.gz" \
  "${LM_LABELS_ROOT}/metrics_index.jsonl.gz"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing label index ${path}" >&2
    exit 1
  fi
done

export PYTHONPATH="${SCRIPTS}${PYTHONPATH:+:${PYTHONPATH}}"
python "${SCRIPTS}/build_curriculum_index.py" \
  --labels-root "${LABELS_ROOT}" \
  --lm-labels-root "${LM_LABELS_ROOT}" \
  --out-dir "${INDEX_DIR}" \
  --parent-layout "${PARENT_LAYOUT}" \
  --parent-version "${PARENT_VERSION}" \
  --parent-manifest-sha256 "${PARENT_MANIFEST_SHA256}"
