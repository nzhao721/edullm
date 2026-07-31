#!/usr/bin/env bash
set -euo pipefail
# Login shell so user exports/profile apply.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  # shellcheck disable=SC1090
  [[ -f "$HOME/.bashrc" ]] && source "$HOME/.bashrc" || true
  [[ -f "$HOME/.profile" ]] && source "$HOME/.profile" || true
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY still unset in login shell" >&2
  exit 2
fi
echo "KEY_OK len=${#WANDB_API_KEY}"
exec bash /mnt/c/alpha_ai/edullm/scripts/farmshare/push_wandb_session_to_farmshare.sh \
  /scratch/users/nzhao2/agent-runs/smollm2-135m-750m-27ep-fresh
