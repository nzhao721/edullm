#!/usr/bin/env bash
# Build learnability-doc filtered corpus from RegMix LM labels.
# Requires finalized labels (READY) unless ALLOW_INCOMPLETE=1.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LABELS_ROOT:?Set LABELS_ROOT to RegMix lm_labels/labels (metrics_index + docs/)}"
: "${WORK:?Set WORK to a writable work directory}"

KEEP_FRACTION="${KEEP_FRACTION:-0.6}"
FILTER_DIR="${FILTER_DIR:-${WORK}/filter}"
CORPUS_DIR="${CORPUS_DIR:-${WORK}/corpus}"
ALLOW_FLAG=()
if [[ "${ALLOW_INCOMPLETE:-0}" == "1" ]]; then
  ALLOW_FLAG=(--allow-incomplete)
fi

mkdir -p "${FILTER_DIR}" "${CORPUS_DIR}"

python "${SCRIPT_DIR}/filter_learnability_docs.py" \
  --labels-root "${LABELS_ROOT}" \
  --out-dir "${FILTER_DIR}" \
  --keep-token-fraction "${KEEP_FRACTION}" \
  "${ALLOW_FLAG[@]}"

python "${SCRIPT_DIR}/build_filtered_corpus.py" \
  --labels-root "${LABELS_ROOT}" \
  --filter-dir "${FILTER_DIR}" \
  --out-dir "${CORPUS_DIR}"

echo "paths_train=${CORPUS_DIR}/paths_train.txt"
echo "corpus_manifest=${CORPUS_DIR}/corpus_manifest.json"
