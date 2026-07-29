#!/usr/bin/env bash
# Deploy task_loss scripts to FarmShare and submit per-checkpoint jobs.
set -Eeuo pipefail

SOCKET="${FARMSHARE_SOCKET:-/tmp/farmshare-nzhao2.sock}"
SUNET="${SUNET:-nzhao2}"
REMOTE="${SUNET}@login.farmshare.stanford.edu"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SCRIPTS="$(cd "${LOCAL_DIR}/.." && pwd)"

remote() {
  ssh -S "${SOCKET}" -o BatchMode=yes "${REMOTE}" "$@"
}

put_file() {
  local src=$1 dst=$2
  remote "mkdir -p $(dirname "${dst}")"
  sed 's/\r$//' "${src}" | remote "cat > ${dst}"
}

STAMP=$(date -u +%Y%m%dT%H%M%S)
RUN_DIR="/scratch/users/${SUNET}/agent-runs/task-loss-per-ckpt-${STAMP}"

echo "Deploying to ${RUN_DIR} ..."
remote "mkdir -p ${RUN_DIR}/scripts ${RUN_DIR}/logs ${RUN_DIR}/eval && chmod 700 ${RUN_DIR}"

put_file "${LOCAL_DIR}/eval_task_loss_olmo_core.py" "${RUN_DIR}/scripts/eval_task_loss_olmo_core.py"
put_file "${LOCAL_DIR}/prepare_model_eval_pt.py" "${RUN_DIR}/scripts/prepare_model_eval_pt.py"
put_file "${LOCAL_DIR}/task_loss_eval_step.sbatch" "${RUN_DIR}/scripts/task_loss_eval_step.sbatch"
put_file "${LOCAL_DIR}/gather_hsdp_step.sbatch" "${RUN_DIR}/scripts/gather_hsdp_step.sbatch"
put_file "${REPO_SCRIPTS}/gather_hsdp_state_pt_to_model.py" "${RUN_DIR}/scripts/gather_hsdp_state_pt_to_model.py"
put_file "${LOCAL_DIR}/submit_per_checkpoint.sh" "${RUN_DIR}/scripts/submit_per_checkpoint.sh"

remote "chmod +x ${RUN_DIR}/scripts/*.sbatch ${RUN_DIR}/scripts/submit_per_checkpoint.sh"

echo "Submitting jobs ..."
remote "SKIP_SCRIPT_COPY=1 RUN_DIR=${RUN_DIR} bash ${RUN_DIR}/scripts/submit_per_checkpoint.sh"
