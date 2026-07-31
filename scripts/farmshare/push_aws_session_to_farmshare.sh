#!/usr/bin/env bash
# Mint AWS session on the engineer laptop (Windows sb-aws-creds) and push
# aws-session.env to one or more FarmShare run directories via the control socket.
#
# Do NOT ask the user to run sb-aws-creds login on FarmShare — it does not work
# there reliably. Mint locally, then upload.
set -Eeuo pipefail

SOCK="${FARMSHARE_SOCK:-/tmp/farmshare-nzhao2.sock}"
HOST="${FARMSHARE_HOST:-nzhao2@login.farmshare.stanford.edu}"
PROFILE="${AWS_PROFILE:-sbsandbox}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PS1_MINT="${SCRIPT_DIR}/mint_aws_session_local.ps1"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_DIR [RUN_DIR...]" >&2
  exit 2
fi

TMPDIR_LOCAL="${TMPDIR:-/tmp}"
LOCAL_ENV="$(mktemp "${TMPDIR_LOCAL}/aws-session.XXXXXX.env")"
cleanup() { rm -f "$LOCAL_ENV"; }
trap cleanup EXIT

# Prefer Windows PowerShell mint (sb-aws-creds works on this device).
if command -v powershell.exe >/dev/null 2>&1; then
  WIN_OUT="$(powershell.exe -NoProfile -Command "[IO.Path]::GetTempFileName()")"
  WIN_OUT="${WIN_OUT//$'\r'/}"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w "$PS1_MINT")" \
    -Profile "$PROFILE" -Region "$REGION" -OutputPath "$WIN_OUT"
  # Copy from Windows temp into WSL temp (binary-safe, no secret echo)
  WIN_WSL="$(wslpath -u "$WIN_OUT")"
  cp -f "$WIN_WSL" "$LOCAL_ENV"
  rm -f "$WIN_WSL"
else
  echo "powershell.exe required to mint via Windows sb-aws-creds" >&2
  exit 1
fi

chmod 600 "$LOCAL_ENV"
# Normalize CRLF from Windows writers
sed -i 's/\r$//' "$LOCAL_ENV"
# Sanity: must contain session token export, must not print values
grep -q '^export AWS_SESSION_TOKEN=' "$LOCAL_ENV"
grep -q '^export AWS_ACCESS_KEY_ID=' "$LOCAL_ENV"

KEY_SUFFIX="$(grep '^export AWS_ACCESS_KEY_ID=' "$LOCAL_ENV" | sed -E "s/.*'([^']+)'/\1/" | tail -c 5)"
echo "local_mint_ok key=...${KEY_SUFFIX}"

for RUN in "$@"; do
  ssh -S "$SOCK" -o BatchMode=yes "$HOST" "mkdir -p '$RUN' && chmod 700 '$RUN'"
  # Stream via SSH (more reliable than scp + ControlPath on this host)
  ssh -S "$SOCK" -o BatchMode=yes "$HOST" \
    "cat > '$RUN/aws-session.env.tmp' && chmod 600 '$RUN/aws-session.env.tmp' && mv -f '$RUN/aws-session.env.tmp' '$RUN/aws-session.env' && chmod 600 '$RUN/aws-session.env'" \
    < "$LOCAL_ENV"
  ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<EOF
set -Eeuo pipefail
set +u
source '$RUN/aws-session.env'
set -u
export PATH="\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}"
aws sts get-caller-identity --query Arn --output text
echo "farmshare_session_ok run=$RUN"
EOF
done
