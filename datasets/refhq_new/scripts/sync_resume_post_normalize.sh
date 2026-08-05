#!/usr/bin/env bash
# Sync documents/-layout fix; wipe partial English outs; resume chain.
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-refhq-new-v1}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/${RUN_NAME}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCAL_PKG="${REPO_ROOT}/datasets/refhq_new"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  "${LOCAL_PKG}/" \
  "${HOST}:${STAGING}/datasets/refhq_new/"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${LOCAL_PKG}/scripts/submit_post_normalize.sh" \
  "${LOCAL_PKG}/scripts/submit_post_holdout.sh" \
  "${HOST}:${RUN_DIR}/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -Eeuo pipefail
STAGING=${STAGING}
RUN_DIR=${RUN_DIR}
find "\${STAGING}/datasets/refhq_new" -type f \\( -name '*.sh' -o -name '*.sbatch' -o -name '*.py' \\) \
  -exec sed -i 's/\r\$//' {} +
sed -i 's/\r\$//' "\${RUN_DIR}/submit_post_normalize.sh" "\${RUN_DIR}/submit_post_holdout.sh"
chmod +x "\${RUN_DIR}/submit_post_normalize.sh" "\${RUN_DIR}/submit_post_holdout.sh"
cp -a "\${STAGING}/datasets/refhq_new" "\${RUN_DIR}/datasets/"
sed -i 's/\r\$//' "\${RUN_DIR}/datasets/refhq_new/scripts/"*.sbatch \
  "\${RUN_DIR}/datasets/refhq_new/scripts/"*.py \
  "\${RUN_DIR}/datasets/refhq_new/process.py"
chmod +x "\${RUN_DIR}/datasets/refhq_new/scripts/"*.sbatch "\${RUN_DIR}/datasets/refhq_new/scripts/"*.sh

# Cancel any leftover refhqn jobs
squeue -u nzhao2 -h -o '%i %j' | awk '/refhqn/ {print \$1}' | xargs -r scancel || true
sleep 2

# Skip login-node dolma smoke (use first array task as canary). Wipe partial outs.
rm -rf "\${RUN_DIR}/out" "\${RUN_DIR}/work" "\${RUN_DIR}/holdout"
mkdir -p "\${RUN_DIR}/out" "\${RUN_DIR}/work" "\${RUN_DIR}/holdout"
rm -f "\${RUN_DIR}/manifests/english_tasks.txt" "\${RUN_DIR}/manifests/tokenize_tasks.txt" \
  "\${RUN_DIR}/manifests/holdout_summary.json" 2>/dev/null || true

source "\${RUN_DIR}/venv/bin/activate"
export PYTHONPATH="\${RUN_DIR}/datasets:\${PYTHONPATH:-}"
source "\${RUN_DIR}/env.sh"
PLAN="\${PLAN:-\${RUN_DIR}/manifests/plan.json}"
REFHQ_NEW_SCRIPTS="\${RUN_DIR}/datasets/refhq_new/scripts"
# Confirm layout contract in source
grep -q 'domain / "documents"' "\${REFHQ_NEW_SCRIPTS}/dolma_english_filter.py"
grep -q 'cwd="/tmp"' "\${RUN_DIR}/datasets/refhq_new/process.py"
echo "layout_and_cwd_fixes_present"

VENV="\${VENV:-\${RUN_DIR}/venv}"
SCRATCH_ROOT="\${SCRATCH_ROOT:-\${RUN_DIR}}"
TOKENIZE_TASKS="\${TOKENIZE_TASKS:-\${SCRATCH_ROOT}/manifests/tokenize_tasks.txt}"
ENGLISH_TASKS="\${ENGLISH_TASKS:-\${SCRATCH_ROOT}/manifests/english_tasks.txt}"
STAGE_DIR="\${STAGE_DIR:-\${RUN_DIR}/publish-stage}"
AWS_SESSION_ENV="\${AWS_SESSION_ENV:-\${RUN_DIR}/aws-session.env}"
SOURCE_LIST="\${SOURCE_LIST:-tulu-v2 openhermes-25 tulu-3 hermes-3 smoltalk dolci}"
SEED="\${SEED:-42}"
S3_BUCKET="\${S3_BUCKET:-edullm-datasets}"
S3_PREFIX="\${S3_PREFIX:-refhq/refhq-new}"

POSTNORM_JOB=\$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:20:00 \
  --job-name=refhqn-postnorm \
  --output=logs/refhqn_post_normalize_%j.out \
  --error=logs/refhqn_post_normalize_%j.err \
  --chdir="\${RUN_DIR}" \
  --export=ALL,RUN_DIR="\${RUN_DIR}",VENV="\${VENV}",PLAN="\${PLAN}",SOURCE_LIST="\${SOURCE_LIST}",SCRATCH_ROOT="\${SCRATCH_ROOT}",REFHQ_NEW_SCRIPTS="\${REFHQ_NEW_SCRIPTS}",STAGE_DIR="\${STAGE_DIR}",S3_BUCKET="\${S3_BUCKET}",S3_PREFIX="\${S3_PREFIX}",SEED="\${SEED}",TOKENIZE_TASKS="\${TOKENIZE_TASKS}",ENGLISH_TASKS="\${ENGLISH_TASKS}",AWS_SESSION_ENV="\${AWS_SESSION_ENV}",SKIP_PUBLISH=0 \
  --wrap="bash \${RUN_DIR}/submit_post_normalize.sh")
echo "resumed_postnorm_job=\${POSTNORM_JOB}"
sleep 10
squeue -u nzhao2 | head -25
REMOTE
