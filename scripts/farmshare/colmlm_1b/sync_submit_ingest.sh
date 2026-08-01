#!/usr/bin/env bash
# Sync Co-LMLM 1B scripts to FarmShare, setup venv, push HF token, submit ingest.
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging/scripts/farmshare/colmlm_1b}"
RUN_NAME="${RUN_NAME:-colmlm-1b-corpus-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
VENV="${VENV:-/scratch/users/${SUNET}/agent-runs/venvs/colmlm-1b}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "mkdir -p '${STAGING}' '${RUN_DIR}/logs' '${RUN_DIR}/data'"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '_remote_probe.sh' \
  "${SRC}/" \
  "${HOST}:${STAGING}/"

# Push HF token if available locally (never echo contents).
TOKEN_FILE=""
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN_FILE="$(mktemp)"
  printf '%s' "${HF_TOKEN}" > "${TOKEN_FILE}"
elif [[ -f /mnt/c/Users/natha/.cache/huggingface/token ]]; then
  TOKEN_FILE="/mnt/c/Users/natha/.cache/huggingface/token"
elif [[ -f "${HOME}/.cache/huggingface/token" ]]; then
  TOKEN_FILE="${HOME}/.cache/huggingface/token"
fi

if [[ -n "${TOKEN_FILE}" ]]; then
  scp -o ControlPath="${SOCK}" -o BatchMode=yes \
    "${TOKEN_FILE}" "${HOST}:${RUN_DIR}/hf_token"
  ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "chmod 600 '${RUN_DIR}/hf_token'"
  echo "pushed hf_token to RUN_DIR"
  if [[ "${TOKEN_FILE}" == /tmp/* ]]; then
    rm -f "${TOKEN_FILE}"
  fi
else
  echo "WARN: no local HF token found; anonymous downloads may hit 429" >&2
fi

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "sed -i 's/\r\$//' '${STAGING}'/*.sh '${STAGING}'/*.py; chmod +x '${STAGING}'/*.sh"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "VENV='${VENV}' bash '${STAGING}/setup_venv.sh'"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "RUN_DIR='${RUN_DIR}' VENV='${VENV}' SCRIPT_DIR='${STAGING}' bash '${STAGING}/submit_ingest.sh'"

echo "submitted ingest for RUN_DIR=${RUN_DIR}"
printf '%s\n' "${RUN_DIR}" > /tmp/colmlm_1b_run_dir.txt
