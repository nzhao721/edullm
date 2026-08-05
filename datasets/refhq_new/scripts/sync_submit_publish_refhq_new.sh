#!/usr/bin/env bash
# Sync only publish/finalize scripts and submit publish against an existing run dir.
# Use when tok/ + manifests already exist under RUN_DIR.
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-refhq-new-v1}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/${RUN_NAME}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPTS_LOCAL="${REPO_ROOT}/datasets/refhq_new/scripts"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "mkdir -p ${STAGING}/datasets/refhq_new/scripts ${STAGING}/datasets ${STAGING}/scripts/farmshare ${RUN_DIR}/logs"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${SCRIPTS_LOCAL}/finalize_upload.py" \
  "${SCRIPTS_LOCAL}/finalize_upload.sbatch" \
  "${SCRIPTS_LOCAL}/publish_refhq_new.py" \
  "${SCRIPTS_LOCAL}/publish_refhq_new.sbatch" \
  "${SCRIPTS_LOCAL}/lib.sh" \
  "${HOST}:${STAGING}/datasets/refhq_new/scripts/"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO_ROOT}/datasets/edullm_text_companion.py" \
  "${REPO_ROOT}/datasets/olmo_shard_utils.py" \
  "${HOST}:${STAGING}/datasets/"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" \
  "${REPO_ROOT}/scripts/farmshare/write_aws_session_env.py" \
  "${HOST}:${STAGING}/scripts/farmshare/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -Eeuo pipefail
STAGING=${STAGING}
RUN_DIR=${RUN_DIR}
mkdir -p "\${RUN_DIR}/datasets/refhq_new/scripts" "\${RUN_DIR}/scripts/farmshare"
cp -a "\${STAGING}/datasets/refhq_new/scripts/." "\${RUN_DIR}/datasets/refhq_new/scripts/"
cp -a "\${STAGING}/datasets/edullm_text_companion.py" "\${RUN_DIR}/datasets/" 2>/dev/null || true
cp -a "\${STAGING}/datasets/olmo_shard_utils.py" "\${RUN_DIR}/datasets/" 2>/dev/null || true
cp -a "\${STAGING}/scripts/farmshare/." "\${RUN_DIR}/scripts/farmshare/"
sed -i 's/\r\$//' "\${RUN_DIR}/datasets/refhq_new/scripts/"*.sh \
  "\${RUN_DIR}/datasets/refhq_new/scripts/"*.py \
  "\${RUN_DIR}/datasets/refhq_new/scripts/"*.sbatch
chmod +x "\${RUN_DIR}/datasets/refhq_new/scripts/"*.sh \
  "\${RUN_DIR}/datasets/refhq_new/scripts/"*.sbatch

PLAN="\${PLAN:-\${RUN_DIR}/manifests/plan.json}"
VENV="\${VENV:-\${RUN_DIR}/venv}"
STAGE_DIR="\${STAGE_DIR:-\${RUN_DIR}/publish-stage}"
AWS_SESSION_ENV="\${AWS_SESSION_ENV:-\${RUN_DIR}/aws-session.env}"
REFHQ_NEW_SCRIPTS="\${RUN_DIR}/datasets/refhq_new/scripts"
export PYTHONPATH="\${RUN_DIR}/datasets:\${PYTHONPATH:-}"

if [[ ! -f "\${AWS_SESSION_ENV}" ]]; then
  echo "ERROR: missing \${AWS_SESSION_ENV}" >&2
  echo "  laptop: scripts/farmshare/push_aws_session_to_farmshare.sh \${RUN_DIR}" >&2
  exit 1
fi

cd "\${RUN_DIR}"
FIN_JOB=\$(sbatch --parsable --exclude=wheat-01 \
  --chdir="\${RUN_DIR}" \
  --export=ALL,RUN_DIR="\${RUN_DIR}",VENV="\${VENV}",PLAN="\${PLAN}",STAGE_DIR="\${STAGE_DIR}",SCRATCH_ROOT="\${RUN_DIR}",REFHQ_NEW_SCRIPTS="\${REFHQ_NEW_SCRIPTS}",AWS_SESSION_ENV="\${AWS_SESSION_ENV}" \
  "\${REFHQ_NEW_SCRIPTS}/finalize_upload.sbatch")
echo "finalize_job=\${FIN_JOB}"

PUB_JOB=\$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterok:\${FIN_JOB} \
  --chdir="\${RUN_DIR}" \
  --export=ALL,RUN_DIR="\${RUN_DIR}",VENV="\${VENV}",PLAN="\${PLAN}",STAGE_DIR="\${STAGE_DIR}",SCRATCH_ROOT="\${RUN_DIR}",REFHQ_NEW_SCRIPTS="\${REFHQ_NEW_SCRIPTS}",AWS_SESSION_ENV="\${AWS_SESSION_ENV}" \
  "\${REFHQ_NEW_SCRIPTS}/publish_refhq_new.sbatch")
echo "publish_job=\${PUB_JOB}"
echo "dataset_id=pretrain/refhq-new"
REMOTE
