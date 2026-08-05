#!/usr/bin/env bash
# Sync publish fix and resubmit publish only (tokens+text already staged).
set -Eeuo pipefail
SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
RUN_DIR="${RUN_DIR:-/scratch/users/nzhao2/refhq-new-v1}"
STAGING="${STAGING:-/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCAL_PKG="${REPO_ROOT}/datasets/refhq_new"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  "${LOCAL_PKG}/" \
  "${HOST}:${STAGING}/datasets/refhq_new/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -Eeuo pipefail
STAGING=${STAGING}
RUN_DIR=${RUN_DIR}
find "\${STAGING}/datasets/refhq_new" -type f \\( -name '*.sh' -o -name '*.sbatch' -o -name '*.py' -o -name '*.md' -o -name '*.yaml' \\) \
  -exec sed -i 's/\r\$//' {} +
cp -a "\${STAGING}/datasets/refhq_new" "\${RUN_DIR}/datasets/"
sed -i 's/\r\$//' "\${RUN_DIR}/datasets/refhq_new/scripts/"*.sbatch "\${RUN_DIR}/datasets/refhq_new/scripts/"*.py
chmod +x "\${RUN_DIR}/datasets/refhq_new/scripts/"*.sbatch "\${RUN_DIR}/datasets/refhq_new/scripts/"*.sh

source "\${RUN_DIR}/env.sh"
VENV="\${VENV:-\${RUN_DIR}/venv}"
PLAN="\${PLAN:-\${RUN_DIR}/manifests/plan.json}"
SCRATCH_ROOT="\${SCRATCH_ROOT:-\${RUN_DIR}}"
STAGE_DIR="\${STAGE_DIR:-\${RUN_DIR}/publish-stage}"
AWS_SESSION_ENV="\${AWS_SESSION_ENV:-\${RUN_DIR}/aws-session.env}"
REFHQ_NEW_SCRIPTS="\${RUN_DIR}/datasets/refhq_new/scripts"
test -d "\${STAGE_DIR}/tokens"
if [[ ! -d "\${STAGE_DIR}/text" && ! -d "\${STAGE_DIR}/vendor" ]]; then
  echo "ERROR: missing text/ or vendor/ under \${STAGE_DIR}" >&2
  exit 1
fi

PUB_JOB=\$(sbatch --parsable --exclude=wheat-01 \
  --chdir="\${RUN_DIR}" \
  --export=ALL,RUN_DIR="\${RUN_DIR}",VENV="\${VENV}",PLAN="\${PLAN}",SCRATCH_ROOT="\${SCRATCH_ROOT}",REFHQ_NEW_SCRIPTS="\${REFHQ_NEW_SCRIPTS}",STAGE_DIR="\${STAGE_DIR}",AWS_SESSION_ENV="\${AWS_SESSION_ENV}",DATASET_ID=pretrain/refhq-instruct \
  "\${REFHQ_NEW_SCRIPTS}/publish_refhq_new.sbatch")
echo "resubmitted_publish_job=\${PUB_JOB} dataset_id=pretrain/refhq-instruct"
squeue -u nzhao2 | head -10
REMOTE
