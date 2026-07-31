#!/usr/bin/env bash
set -euo pipefail
VENV=/scratch/users/nzhao2/agent-runs/venvs/smollm2-train
source "${VENV}/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q 'wandb>=0.17'
python -c "import wandb; print('wandb', wandb.__version__)"
