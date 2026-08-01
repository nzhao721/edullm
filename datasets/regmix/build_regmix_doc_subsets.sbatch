#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Filtering is memory-bound; the middle-PPL tokenizer can use all eight cores.
#SBATCH --cpus-per-task=8
# The middle-PPL materializer groups selected source documents before tokenizing.
# 64 GiB: the 32 GiB job OOM'd at ~32 GiB RSS during middle-PPL corpus build.
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=regmix-doc-subsets
#SBATCH --output=logs/regmix_doc_subsets_%j.out
#SBATCH --error=logs/regmix_doc_subsets_%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${REGMIX_ROOT:?}" "${SCRIPTS:?}" "${WORK_DIR:?}"

# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"

LABELS_ROOT="${LABELS_ROOT:-${REGMIX_ROOT}/lm_labels/labels}"
for path in "${LABELS_ROOT}/READY" "${LABELS_ROOT}/metrics_index.jsonl.gz"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing finalized LM-label artifact ${path}" >&2
    exit 1
  fi
done

LEARN_DIR="${WORK_DIR}/learnability-doc-top60"
MIDDLE_DIR="${WORK_DIR}/middle-ppl-doc-mid60"
mkdir -p "${LEARN_DIR}" "${MIDDLE_DIR}"

LEARN_MANIFEST="${LEARN_DIR}/corpus/corpus_manifest.json"
MIDDLE_MANIFEST="${MIDDLE_DIR}/corpus/corpus_manifest.json"
MIDDLE_FILTER_SUMMARY="${MIDDLE_DIR}/filter/filter_summary.json"
# Middle-PPL tokenization is memory-bound; cap parallelism below Slurm CPU count.
MIDDLE_WORKERS="${MIDDLE_WORKERS:-4}"
MIDDLE_BATCH_SIZE="${MIDDLE_BATCH_SIZE:-64}"

if [[ -f "${LEARN_MANIFEST}" ]]; then
  echo "SKIP learnability filter+build (found ${LEARN_MANIFEST})"
else
  python "${SCRIPTS}/filter_learnability_docs.py" \
    --labels-root "${LABELS_ROOT}" \
    --out-dir "${LEARN_DIR}/filter" \
    --keep-token-fraction 0.6
  python "${SCRIPTS}/build_learnability_corpus.py" \
    --labels-root "${LABELS_ROOT}" \
    --filter-dir "${LEARN_DIR}/filter" \
    --out-dir "${LEARN_DIR}/corpus"
fi

if [[ -f "${MIDDLE_MANIFEST}" ]]; then
  echo "SKIP middle-PPL filter+build (found ${MIDDLE_MANIFEST})"
elif [[ -f "${MIDDLE_FILTER_SUMMARY}" ]]; then
  echo "SKIP middle-PPL filter (found ${MIDDLE_FILTER_SUMMARY})"
  python "${SCRIPTS}/build_middle_ppl_corpus.py" \
    --labels-root "${LABELS_ROOT}" \
    --keep-manifest "${MIDDLE_DIR}/filter/keep_manifest.jsonl.gz" \
    --keep-ids "${MIDDLE_DIR}/filter/keep_ids.txt" \
    --out-dir "${MIDDLE_DIR}/corpus" \
    --workers "${MIDDLE_WORKERS}" \
    --batch-size "${MIDDLE_BATCH_SIZE}"
else
  python "${SCRIPTS}/filter_middle_ppl_docs.py" \
    --labels-root "${LABELS_ROOT}" \
    --out-dir "${MIDDLE_DIR}/filter" \
    --keep-frac 0.6 \
    --metric avg_perplexity
  python "${SCRIPTS}/build_middle_ppl_corpus.py" \
    --labels-root "${LABELS_ROOT}" \
    --keep-manifest "${MIDDLE_DIR}/filter/keep_manifest.jsonl.gz" \
    --keep-ids "${MIDDLE_DIR}/filter/keep_ids.txt" \
    --out-dir "${MIDDLE_DIR}/corpus" \
    --workers "${MIDDLE_WORKERS}" \
    --batch-size "${MIDDLE_BATCH_SIZE}"
fi
