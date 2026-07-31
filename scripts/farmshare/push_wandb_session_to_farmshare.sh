#!/usr/bin/env bash
# Push WANDB_API_KEY into FarmShare scratch. Never prints the key.
# Source order: $WANDB_API_KEY env, then KEY_FILE (default ~/.wandb_api_key or Windows path).
set -euo pipefail
SOCKET="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
SUNET="${SUNET:-nzhao2}"
HOST="${HOST:-${SUNET}@login.farmshare.stanford.edu}"
DEST="${1:?usage: push_wandb_session_to_farmshare.sh RUN_DIR [KEY_FILE]}"
KEY_FILE="${2:-}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  for candidate in \
    "${KEY_FILE}" \
    "${HOME}/.wandb_api_key" \
    "/mnt/c/Users/natha/.wandb_api_key" \
    "/mnt/c/Users/${USER}/.wandb_api_key"
  do
    [[ -n "${candidate}" && -f "${candidate}" ]] || continue
    WANDB_API_KEY="$(tr -d ' \t\r\n' < "${candidate}")"
    export WANDB_API_KEY
    break
  done
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY not found in env or key file" >&2
  echo "Create /mnt/c/Users/natha/.wandb_api_key with the key (one line), then retry." >&2
  exit 2
fi

TMP="$(mktemp)"
chmod 600 "${TMP}"
cat > "${TMP}" <<EOF
# Generated for FarmShare Slurm jobs. Do not commit.
export WANDB_API_KEY='${WANDB_API_KEY}'
export WANDB_START_METHOD=thread
EOF

ssh -S "${SOCKET}" -o BatchMode=yes "${HOST}" "mkdir -p '${DEST}' && chmod 700 '${DEST}'"
scp -o ControlPath="${SOCKET}" "${TMP}" "${HOST}:${DEST}/wandb-session.env"
ssh -S "${SOCKET}" -o BatchMode=yes "${HOST}" "chmod 600 '${DEST}/wandb-session.env' && wc -c '${DEST}/wandb-session.env' | awk '{print \"wrote_bytes\", \$1}'"
rm -f "${TMP}"
echo "pushed wandb-session.env -> ${DEST}/wandb-session.env"
