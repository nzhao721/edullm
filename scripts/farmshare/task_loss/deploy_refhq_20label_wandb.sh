#!/usr/bin/env bash
# Deploy RefHQ 20-label re-eval + W&B logger to FarmShare and submit.
# Requires an open FarmShare control socket.
set -Eeuo pipefail

SOCKET="${FARMSHARE_SOCKET:-/tmp/farmshare-nzhao2.sock}"
SUNET="${SUNET:-nzhao2}"
REMOTE="${SUNET}@login.farmshare.stanford.edu"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FS_SCRIPTS="$(cd "${LOCAL_DIR}/.." && pwd)"

if [[ ! -S "${SOCKET}" ]]; then
  echo "FarmShare control socket missing: ${SOCKET}" >&2
  echo "Open it locally first:" >&2
  echo "  ssh -M -S ${SOCKET} -o ControlPersist=yes ${REMOTE}" >&2
  exit 2
fi

remote() {
  ssh -S "${SOCKET}" -o BatchMode=yes "${REMOTE}" "$@"
}

put_file() {
  local src=$1 dst=$2
  remote "mkdir -p $(dirname "${dst}")"
  sed 's/\r$//' "${src}" | remote "cat > ${dst}"
}

STAMP=$(date -u +%Y%m%dT%H%M%S)
RUN_DIR="/scratch/users/${SUNET}/agent-runs/refhq-20label-wandb-${STAMP}"

echo "Deploying to ${RUN_DIR} ..."
remote "mkdir -p ${RUN_DIR}/scripts ${RUN_DIR}/logs ${RUN_DIR}/eval/refhq && chmod 700 ${RUN_DIR}"

put_file "${LOCAL_DIR}/eval_task_loss_olmo_core.py" "${RUN_DIR}/scripts/eval_task_loss_olmo_core.py"
put_file "${LOCAL_DIR}/prepare_model_eval_pt.py" "${RUN_DIR}/scripts/prepare_model_eval_pt.py"
put_file "${LOCAL_DIR}/task_loss_eval_step.sbatch" "${RUN_DIR}/scripts/task_loss_eval_step.sbatch"
put_file "${LOCAL_DIR}/log_refhq_task_loss_to_wandb.py" "${RUN_DIR}/scripts/log_refhq_task_loss_to_wandb.py"
put_file "${LOCAL_DIR}/submit_refhq_20label_wandb.sh" "${RUN_DIR}/scripts/submit_refhq_20label_wandb.sh"

remote "chmod +x ${RUN_DIR}/scripts/*.sbatch ${RUN_DIR}/scripts/submit_refhq_20label_wandb.sh"

echo "Pushing wandb-session.env ..."
bash "${FS_SCRIPTS}/push_wandb_session_to_farmshare.sh" "${RUN_DIR}"

# Optional HF token for OLMES dataset / tokenizer downloads.
if [[ -f "${FS_SCRIPTS}/push_hf_session_to_farmshare.sh" ]]; then
  bash "${FS_SCRIPTS}/push_hf_session_to_farmshare.sh" "${RUN_DIR}" || true
elif [[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" || -f "${HOME}/.hf_token" || -f "/mnt/c/Users/natha/.hf_token" ]]; then
  # Minimal HF session push without a dedicated helper.
  KEY=""
  if [[ -n "${HF_TOKEN:-}" ]]; then KEY="${HF_TOKEN}"
  elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then KEY="${HUGGING_FACE_HUB_TOKEN}"
  elif [[ -f "${HOME}/.hf_token" ]]; then KEY="$(tr -d ' \t\r\n' < "${HOME}/.hf_token")"
  elif [[ -f "/mnt/c/Users/natha/.hf_token" ]]; then KEY="$(tr -d ' \t\r\n' < /mnt/c/Users/natha/.hf_token)"
  fi
  if [[ -n "${KEY}" ]]; then
    TMP="$(mktemp)"
    chmod 600 "${TMP}"
    printf "export HF_TOKEN='%s'\nexport HUGGING_FACE_HUB_TOKEN='%s'\n" "${KEY}" "${KEY}" > "${TMP}"
    scp -o ControlPath="${SOCKET}" "${TMP}" "${REMOTE}:${RUN_DIR}/hf-session.env"
    remote "chmod 600 '${RUN_DIR}/hf-session.env'"
    rm -f "${TMP}"
    echo "pushed hf-session.env"
  fi
fi

echo "Submitting RefHQ 20-label evals + W&B log job ..."
remote "SKIP_SCRIPT_COPY=1 RUN_DIR=${RUN_DIR} WANDB_PROJECT=refhq WANDB_ENTITY=eduLLM bash ${RUN_DIR}/scripts/submit_refhq_20label_wandb.sh"

echo "DONE RUN_DIR=${RUN_DIR}"
