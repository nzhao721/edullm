#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY=set len=${#WANDB_API_KEY}"
else
  echo "WANDB_API_KEY=unset"
fi
for f in "$HOME/.bashrc" "$HOME/.profile" "$HOME/.zshrc" /mnt/c/Users/natha/.bashrc; do
  [[ -f "$f" ]] && grep -n WANDB "$f" 2>/dev/null | sed 's/=.*/=*** /' || true
done
# Windows user env via powershell without printing value
powershell.exe -NoProfile -Command "if ([string]::IsNullOrEmpty(\$env:WANDB_API_KEY)) { 'win_WANDB_API_KEY=unset' } else { 'win_WANDB_API_KEY=set' }"
