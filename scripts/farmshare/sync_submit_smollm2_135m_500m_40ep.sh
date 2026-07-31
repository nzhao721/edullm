#!/usr/bin/env bash
# Sync SmolLM2 500M/40-epoch training scripts to FarmShare and submit a 2x2-GPU job.
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "mkdir -p ${STAGING}/scripts/farmshare"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${SRC}/scripts/farmshare/slice_tokenized_subset.py" \
  "${SRC}/scripts/farmshare/eval_arc_task_loss_smollm.py" \
  "${SRC}/scripts/farmshare/train_smollm2_135m_ddp.py" \
  "${SRC}/scripts/farmshare/setup_smollm2_train_venv.sh" \
  "${SRC}/scripts/farmshare/submit_smollm2_135m_500m_40ep.sh" \
  "${HOST}:${STAGING}/scripts/farmshare/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<'REMOTE'
set -Eeuo pipefail
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
sed -i 's/\r$//' "${STAGING}/scripts/farmshare/"*.sh "${STAGING}/scripts/farmshare/"*.py
chmod +x "${STAGING}/scripts/farmshare/"*.sh
bash "${STAGING}/scripts/farmshare/setup_smollm2_train_venv.sh"
REMOTE

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -Eeuo pipefail
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
SRC_DATA=/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-tokenized
DATA_DIR=/scratch/users/nzhao2/agent-runs/fineweb-edu-500m-smollm2-tokenized
VENV=/scratch/users/nzhao2/agent-runs/venvs/smollm2-train
SLICE_PY="\${STAGING}/scripts/farmshare/slice_tokenized_subset.py"
if [[ ! -f "\${DATA_DIR}/meta.json" ]]; then
  source "\${VENV}/bin/activate"
  python -u "\${SLICE_PY}" --src-dir "\${SRC_DATA}" --dst-dir "\${DATA_DIR}" --max-tokens 500000000
fi
REMOTE

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash "${STAGING}/scripts/farmshare/submit_smollm2_135m_500m_40ep.sh"
