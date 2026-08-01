#!/usr/bin/env bash
# Sync the CPU-only RegMix document-subset build to FarmShare and submit it.
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REGMIX_LOCAL="${REPO_ROOT}/datasets/regmix"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "mkdir -p '${STAGING}/datasets/regmix' '${STAGING}/experiments/token-selection/learnability-doc' '${STAGING}/experiments/token-selection/middle-ppl-doc'"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REGMIX_LOCAL}/build_regmix_doc_subsets.sbatch" \
  "${REGMIX_LOCAL}/publish_regmix_doc_subset_edullm_data.sbatch" \
  "${REGMIX_LOCAL}/publish_regmix_doc_subset_edullm_data.py" \
  "${REGMIX_LOCAL}/publish_regmix_edullm_data.py" \
  "${REGMIX_LOCAL}/submit_build_regmix_doc_subsets.sh" \
  "${HOST}:${STAGING}/datasets/regmix/"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO_ROOT}/experiments/token-selection/learnability-doc/filter_learnability_docs.py" \
  "${REPO_ROOT}/experiments/token-selection/learnability-doc/build_filtered_corpus.py" \
  "${HOST}:${STAGING}/experiments/token-selection/learnability-doc/"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO_ROOT}/experiments/token-selection/middle-ppl-doc/filter_middle_ppl_docs.py" \
  "${REPO_ROOT}/experiments/token-selection/middle-ppl-doc/build_filtered_corpus.py" \
  "${HOST}:${STAGING}/experiments/token-selection/middle-ppl-doc/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -Eeuo pipefail
STAGING=${STAGING}
sed -i 's/\r\$//' "\${STAGING}/datasets/regmix/"*.sh "\${STAGING}/datasets/regmix/"*.py "\${STAGING}/datasets/regmix/"*.sbatch
chmod +x "\${STAGING}/datasets/regmix/"*.sh
bash "\${STAGING}/datasets/regmix/submit_build_regmix_doc_subsets.sh"
REMOTE
