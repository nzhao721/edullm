#!/usr/bin/env bash
# Sync repair/layout/curriculum scripts and submit stream n_tokens rebuild.
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REGMIX_LOCAL="${REPO_ROOT}/datasets/regmix"
CURRICULUM_LOCAL="${REPO_ROOT}/experiments/curriculum"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/regmix-stream-ntokens-$(date -u +%Y%m%dT%H%M%SZ)}"
DOMAINS="${DOMAINS:-algebraic-stack arxiv dclm open-web-math pes2o starcoder wiki}"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "mkdir -p '${STAGING}/datasets/regmix' '${STAGING}/experiments/curriculum/scripts' '${RUN_DIR}' && chmod 700 '${RUN_DIR}'"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REGMIX_LOCAL}/rebuild_stream_n_tokens.py" \
  "${REGMIX_LOCAL}/rebuild_stream_n_tokens.sbatch" \
  "${REGMIX_LOCAL}/submit_rebuild_stream_n_tokens.sh" \
  "${REGMIX_LOCAL}/capture_regmix_parent_layout.py" \
  "${REGMIX_LOCAL}/publish_regmix_curriculum_edullm_data.py" \
  "${REGMIX_LOCAL}/publish_regmix_curriculum_edullm_data.sbatch" \
  "${REGMIX_LOCAL}/submit_publish_regmix_curriculum_edullm_data.sh" \
  "${REGMIX_LOCAL}/build_regmix_curriculum_index.sbatch" \
  "${HOST}:${STAGING}/datasets/regmix/"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${CURRICULUM_LOCAL}/curriculum_pacing.py" \
  "${HOST}:${STAGING}/experiments/curriculum/"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${CURRICULUM_LOCAL}/scripts/build_curriculum_index.py" \
  "${HOST}:${STAGING}/experiments/curriculum/scripts/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -Eeuo pipefail
STAGING=${STAGING}
RUN_DIR=${RUN_DIR}
DOMAINS='${DOMAINS}'
sed -i 's/\r\$//' "\${STAGING}/datasets/regmix/"*.sh "\${STAGING}/datasets/regmix/"*.py "\${STAGING}/datasets/regmix/"*.sbatch \
  "\${STAGING}/experiments/curriculum/"*.py "\${STAGING}/experiments/curriculum/scripts/"*.py || true
chmod +x "\${STAGING}/datasets/regmix/"*.sh
RUN_DIR="\${RUN_DIR}" DOMAINS="\${DOMAINS}" \
  bash "\${STAGING}/datasets/regmix/submit_rebuild_stream_n_tokens.sh"
REMOTE
