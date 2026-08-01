#!/usr/bin/env bash
# Resume a failed doc-subset build (skips completed learnability + middle filter).
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/regmix-doc-subsets-20260731T221030Z}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REGMIX_LOCAL="${REPO_ROOT}/datasets/regmix"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REGMIX_LOCAL}/build_regmix_doc_subsets.sbatch" \
  "${REGMIX_LOCAL}/submit_build_regmix_doc_subsets.sh" \
  "${HOST}:${STAGING}/datasets/regmix/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -Eeuo pipefail
STAGING=${STAGING}
RUN_DIR=${RUN_DIR}
sed -i 's/\r\$//' "\${STAGING}/datasets/regmix/build_regmix_doc_subsets.sbatch" \
  "\${STAGING}/datasets/regmix/submit_build_regmix_doc_subsets.sh"
chmod +x "\${STAGING}/datasets/regmix/submit_build_regmix_doc_subsets.sh"
RESUME=1 RUN_DIR="\${RUN_DIR}" WORK_DIR="\${RUN_DIR}/work" \
  bash "\${STAGING}/datasets/regmix/submit_build_regmix_doc_subsets.sh"
REMOTE
