#!/usr/bin/env bash
# Resubmit FineWeb 1B publish into an existing RUN_DIR (reuse venv + edullm-data).
set -Eeuo pipefail
SUNET="${SUNET:-nzhao2}"
SOCK="${FARMSHARE_SOCK:-/tmp/farmshare-${SUNET}.sock}"
HOST="${SUNET}@login.farmshare.stanford.edu"
EDULLM_ROOT="${EDULLM_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
RUN_DIR="${1:?usage: $0 RUN_DIR}"
STAGE_DIR="${STAGE_DIR:-${RUN_DIR}/publish-stage}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

sed -i 's/\r$//' "${REPO_ROOT}/datasets/fineweb/"* || true
rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO_ROOT}/datasets/fineweb/" \
  "${HOST}:${EDULLM_ROOT}/datasets/fineweb/"

bash "${REPO_ROOT}/scripts/farmshare/push_aws_session_to_farmshare.sh" "${RUN_DIR}"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<EOF
set -Eeuo pipefail
mkdir -p '${RUN_DIR}/logs'
cd '${RUN_DIR}'
sbatch --exclude=wheat-01 \
  --export=ALL,RUN_DIR='${RUN_DIR}',STAGE_DIR='${STAGE_DIR}',EDULLM_ROOT='${EDULLM_ROOT}',AWS_SESSION_ENV='${RUN_DIR}/aws-session.env',SKIP_EDULLM_DATA_INSTALL=1,SKIP_STAGE='${SKIP_STAGE:-0}',SKIP_TOKENIZER_PUBLISH='${SKIP_TOKENIZER_PUBLISH:-0}' \
  '${EDULLM_ROOT}/datasets/fineweb/publish_fineweb_edu_1b_edullm_data.sbatch'
EOF
