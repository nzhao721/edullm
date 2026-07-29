#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=refhq-unshard
#SBATCH --output=logs/refhq_unshard_%j.out
#SBATCH --error=logs/refhq_unshard_%j.err

set -Eeuo pipefail
: "${RUN_DIR:?Set RUN_DIR to scratch run directory}"
: "${VENV_ACTIVATE:=/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/venv/bin/activate}"

cd "$RUN_DIR"
mkdir -p logs
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi
bash "$RUN_DIR/unshard_refhq_step1315.sh"
