#!/usr/bin/env bash
# After stream lengths exist: merge repaired LM index, build curriculum ranks, publish.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
REGMIX_ROOT="${REGMIX_ROOT:-/scratch/users/${SUNET}/agent-runs/regmix-10b-20260725-124810}"
STAGING_ROOT="${STAGING_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
NTOK_RUN="${NTOK_RUN:?set NTOK_RUN to stream-ntokens run dir}"
INDEX_DIR="${INDEX_DIR:-${REGMIX_ROOT}/curriculum_index}"
PARENT_LAYOUT="${PARENT_LAYOUT:-${NTOK_RUN}/parent_layout.json}"
PARENT_VERSION="${PARENT_VERSION:-v1}"
PARENT_MANIFEST_SHA256="${PARENT_MANIFEST_SHA256:-a24992f53dc4a900bacf8fa571d77e343fd28ffa9054c14b93d54204b0a38cb4}"
PUBLISH_RUN="${PUBLISH_RUN:-/scratch/users/${SUNET}/agent-runs/regmix-curriculum-edullm-publish-$(date -u +%Y%m%dT%H%M%SZ)}"
REPAIRED_LM_ROOT="${REPAIRED_LM_ROOT:-${NTOK_RUN}/lm_labels_stream_aligned}"
REPAIRED_INDEX="${REPAIRED_LM_ROOT}/metrics_index.jsonl.gz"

mkdir -p "${NTOK_RUN}/logs" "${REPAIRED_LM_ROOT}" "${INDEX_DIR}" "${PUBLISH_RUN}/logs" \
  "${PUBLISH_RUN}/scripts/regmix" "${PUBLISH_RUN}/scripts/curriculum"
cd "${NTOK_RUN}"

missing=0
for domain in algebraic-stack arxiv dclm open-web-math pes2o starcoder wiki; do
  if [[ ! -f "${NTOK_RUN}/stream_lengths/${domain}.n_tokens.json" ]]; then
    echo "ERROR: missing ${NTOK_RUN}/stream_lengths/${domain}.n_tokens.json" >&2
    missing=1
  fi
done
[[ "${missing}" -eq 0 ]] || exit 1

if [[ -d "${REGMIX_ROOT}/venv" && ! -e "${NTOK_RUN}/venv" ]]; then
  ln -s "${REGMIX_ROOT}/venv" "${NTOK_RUN}/venv"
fi
# shellcheck disable=SC1091
source "${NTOK_RUN}/venv/bin/activate"

cp -a "${STAGING_ROOT}/datasets/regmix/rebuild_stream_n_tokens.py" "${NTOK_RUN}/scripts/" 2>/dev/null || \
  cp -a "${STAGING_ROOT}/datasets/regmix/rebuild_stream_n_tokens.py" "${NTOK_RUN}/scripts/rebuild_stream_n_tokens.py"
mkdir -p "${NTOK_RUN}/scripts"
cp -a "${STAGING_ROOT}/datasets/regmix/"*.py "${NTOK_RUN}/scripts/" || true
cp -a "${STAGING_ROOT}/datasets/regmix/"*.sbatch "${NTOK_RUN}/scripts/" || true
cp -a "${STAGING_ROOT}/experiments/curriculum/curriculum_pacing.py" "${NTOK_RUN}/scripts/"
cp -a "${STAGING_ROOT}/experiments/curriculum/scripts/build_curriculum_index.py" "${NTOK_RUN}/scripts/"
sed -i 's/\r$//' "${NTOK_RUN}/scripts/"*.py "${NTOK_RUN}/scripts/"*.sbatch || true

if [[ ! -f "${PARENT_LAYOUT}" ]]; then
  python "${NTOK_RUN}/scripts/capture_regmix_parent_layout.py" \
    --tokenized-root "${REGMIX_ROOT}/tokenized" \
    --out "${PARENT_LAYOUT}"
fi

echo "=== merge repaired LM metrics index ==="
python "${NTOK_RUN}/scripts/rebuild_stream_n_tokens.py" \
  --regmix-root "${REGMIX_ROOT}" \
  --skip-retokenize \
  --lengths-dir "${NTOK_RUN}/stream_lengths" \
  --out-index "${REPAIRED_INDEX}"

# Point a labels-compatible root at the repaired index for the builder.
mkdir -p "${REPAIRED_LM_ROOT}"
# Builder expects metrics_index.jsonl.gz under --lm-labels-root
if [[ ! -f "${REPAIRED_INDEX}" ]]; then
  echo "ERROR: repaired index missing" >&2
  exit 1
fi

echo "=== submit curriculum index build ==="
INDEX_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --chdir="${NTOK_RUN}" \
  --job-name=regmix-curr-index \
  --export=ALL,RUN_DIR="${NTOK_RUN}",REGMIX_ROOT="${REGMIX_ROOT}",INDEX_DIR="${INDEX_DIR}",SCRIPTS="${NTOK_RUN}/scripts",PARENT_LAYOUT="${PARENT_LAYOUT}",PARENT_VERSION="${PARENT_VERSION}",PARENT_MANIFEST_SHA256="${PARENT_MANIFEST_SHA256}",LABELS_ROOT="${REGMIX_ROOT}/labels",LM_LABELS_ROOT="${REPAIRED_LM_ROOT}" \
  "${STAGING_ROOT}/datasets/regmix/build_regmix_curriculum_index.sbatch")
echo "index_job=${INDEX_JOB}"
echo "index_dir=${INDEX_DIR}"
echo "repaired_lm_root=${REPAIRED_LM_ROOT}"
echo "publish_run_pending=${PUBLISH_RUN}"
echo "NOTE: publish after index job completes successfully"
