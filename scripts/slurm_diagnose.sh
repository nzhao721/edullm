#!/bin/bash
#SBATCH --job-name=edullm-diagnose
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --chdir=/home/users/nzhao2/edullm
#SBATCH --output=logs/diagnose_%j.out
#SBATCH --error=logs/diagnose_%j.err

set -eo pipefail

echo "hostname=$(hostname)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "PWD=$(pwd)"
echo "farmshare.env exists? $([[ -f farmshare.env ]] && echo yes || echo no)"
echo "slurm_common exists? $([[ -f scripts/slurm_common.sh ]] && echo yes || echo no)"
echo "venv exists? $([[ -f .venv/bin/activate ]] && echo yes || echo no)"

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-/home/users/nzhao2/edullm}/scripts/slurm_common.sh"
slurm_project_setup

python - <<'PY'
import sys
print("python executable:", sys.executable)
import torch
print("torch:", torch.__version__)
print("diagnose ok")
PY
