#!/usr/bin/env bash
# Sync refhq_new package + scripts to FarmShare staging and submit the full chain.
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-refhq-new-v1}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/${RUN_NAME}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCAL_PKG="${REPO_ROOT}/datasets/refhq_new"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "mkdir -p ${STAGING}/datasets/refhq_new/scripts ${STAGING}/datasets"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  "${LOCAL_PKG}/" \
  "${HOST}:${STAGING}/datasets/refhq_new/"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO_ROOT}/datasets/olmo_shard_utils.py" \
  "${REPO_ROOT}/datasets/trim_and_tokenize_regmix.py" \
  "${REPO_ROOT}/datasets/edullm_text_companion.py" \
  "${HOST}:${STAGING}/datasets/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -Eeuo pipefail
STAGING=${STAGING}
RUN_DIR=${RUN_DIR}
SUNET=${SUNET}
RUN_NAME=${RUN_NAME}
mkdir -p "\${RUN_DIR}/datasets" "\${RUN_DIR}/logs"
# Normalize CRLF from Windows rsync
find "\${STAGING}/datasets/refhq_new" -type f \\( -name '*.sh' -o -name '*.sbatch' -o -name '*.py' \\) \
  -exec sed -i 's/\r\$//' {} +
chmod +x "\${STAGING}/datasets/refhq_new/scripts/"*.sh "\${STAGING}/datasets/refhq_new/scripts/"*.sbatch 2>/dev/null || true
export SUNET RUN_NAME RUN_DIR STAGING_ROOT="\${STAGING}"
bash "\${STAGING}/datasets/refhq_new/scripts/submit_refhq_new.sh"
REMOTE
