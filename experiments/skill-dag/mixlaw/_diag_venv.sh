#!/usr/bin/env bash
set -Eeuo pipefail
ssh -S /tmp/farmshare-nzhao2.sock -o BatchMode=yes nzhao2@login.farmshare.stanford.edu bash -s <<'REMOTE'
set -Eeuo pipefail
ls -d /scratch/users/nzhao2/agent-runs/olmohq-topup-2026* 2>/dev/null || true
RUN=$(ls -1d /scratch/users/nzhao2/agent-runs/olmohq-topup-2026* 2>/dev/null | tail -1)
echo "RUN=$RUN"
if [[ -n "${RUN:-}" ]]; then
  "$RUN/venv/bin/python" -c 'import sys; print(sys.executable); import huggingface_hub; print(huggingface_hub.__version__)' || echo 'import_failed'
  "$RUN/venv/bin/pip" show huggingface-hub | head -5 || true
fi
REMOTE
