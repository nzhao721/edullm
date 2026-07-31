#!/usr/bin/env bash
set -euo pipefail
scancel 1670807 2>/dev/null || true
squeue -u nzhao2 | head -20
PY=/scratch/users/nzhao2/agent-runs/venvs/smollm2-train/bin/python
PIP=/scratch/users/nzhao2/agent-runs/venvs/smollm2-train/bin/pip
"$PIP" show wandb 2>/dev/null | head -3 || echo "wandb_not_installed"
"$PY" - <<'PY'
import os
print("WANDB_API_KEY", "set" if os.environ.get("WANDB_API_KEY") else "unset")
print("HOME_netrc", os.path.exists(os.path.expanduser("~/.netrc")))
PY
ls -la /scratch/users/nzhao2/.netrc 2>/dev/null || true
ls -la /home/users/nzhao2/.netrc 2>/dev/null || true
