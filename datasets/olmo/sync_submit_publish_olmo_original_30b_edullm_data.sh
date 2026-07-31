#!/usr/bin/env bash
# Sync olmo-original-30b publish scripts to FarmShare and submit.
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "mkdir -p ${STAGING}/datasets/olmo ${STAGING}/datasets/olmohq ${STAGING}/scripts/farmshare"

# Shared publisher (updated) + 30b wrappers.
rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO_ROOT}/datasets/olmohq/publish_olmohq_edullm_data.py" \
  "${HOST}:${STAGING}/datasets/olmohq/"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO_ROOT}/datasets/olmo/publish_olmo_original_30b_edullm_data.sbatch" \
  "${REPO_ROOT}/datasets/olmo/submit_publish_olmo_original_30b_edullm_data.sh" \
  "${HOST}:${STAGING}/datasets/olmo/"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" \
  "${REPO_ROOT}/scripts/farmshare/write_aws_session_env.py" \
  "${HOST}:${STAGING}/scripts/farmshare/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -Eeuo pipefail
STAGING=${STAGING}
sed -i 's/\r\$//' \
  "\${STAGING}/datasets/olmohq/"*.py \
  "\${STAGING}/datasets/olmo/"*.sh \
  "\${STAGING}/datasets/olmo/"*.sbatch
sed -i 's/self\.out_dir = out_di\$/self.out_dir = out_dir/' \
  "\${STAGING}/datasets/olmohq/publish_olmohq_edullm_data.py"
chmod +x "\${STAGING}/datasets/olmo/"*.sh
bash "\${STAGING}/datasets/olmo/submit_publish_olmo_original_30b_edullm_data.sh"
REMOTE
