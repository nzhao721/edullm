#!/usr/bin/env bash
# Build both finalized-label RegMix document subsets on FarmShare (no AWS writes).
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
REGMIX_ROOT="${REGMIX_ROOT:-/scratch/users/${SUNET}/agent-runs/regmix-10b-20260725-124810}"
STAGING_ROOT="${STAGING_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/regmix-doc-subsets-$(date -u +%Y%m%dT%H%M%SZ)}"
WORK_DIR="${WORK_DIR:-${RUN_DIR}/work}"
RESUME="${RESUME:-0}"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/scripts/regmix"
cd "${RUN_DIR}"
if [[ "${RESUME}" == "1" ]]; then
  echo "resume=1 run_dir=${RUN_DIR}"
fi

if [[ -d "${REGMIX_ROOT}/venv" && ! -e "${RUN_DIR}/venv" ]]; then
  ln -s "${REGMIX_ROOT}/venv" "${RUN_DIR}/venv"
fi
if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  echo "ERROR: missing RegMix Python environment at ${REGMIX_ROOT}/venv" >&2
  exit 1
fi

for file in \
  build_regmix_doc_subsets.sbatch \
  publish_regmix_doc_subset_edullm_data.sbatch \
  publish_regmix_doc_subset_edullm_data.py \
  publish_regmix_edullm_data.py; do
  cp -a "${STAGING_ROOT}/datasets/regmix/${file}" "${RUN_DIR}/scripts/regmix/"
done
cp -a "${STAGING_ROOT}/experiments/token-selection/learnability-doc/filter_learnability_docs.py" \
  "${RUN_DIR}/scripts/regmix/"
cp -a "${STAGING_ROOT}/experiments/token-selection/learnability-doc/build_filtered_corpus.py" \
  "${RUN_DIR}/scripts/regmix/build_learnability_corpus.py"
cp -a "${STAGING_ROOT}/experiments/token-selection/middle-ppl-doc/filter_middle_ppl_docs.py" \
  "${RUN_DIR}/scripts/regmix/"
cp -a "${STAGING_ROOT}/experiments/token-selection/middle-ppl-doc/build_filtered_corpus.py" \
  "${RUN_DIR}/scripts/regmix/build_middle_ppl_corpus.py"
sed -i 's/\r$//' "${RUN_DIR}/scripts/regmix/"*.py "${RUN_DIR}/scripts/regmix/"*.sbatch

JOB=$(sbatch --parsable --exclude=wheat-01 \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",REGMIX_ROOT="${REGMIX_ROOT}",WORK_DIR="${WORK_DIR}",SCRIPTS="${RUN_DIR}/scripts/regmix" \
  "${RUN_DIR}/scripts/regmix/build_regmix_doc_subsets.sbatch")
echo "build_job=${JOB}"
echo "run_dir=${RUN_DIR}"
echo "work_dir=${WORK_DIR}"
echo "learnability_dataset=pretrain/learnability-doc-top60"
echo "middle_ppl_dataset=pretrain/middle-ppl-doc-mid60"
