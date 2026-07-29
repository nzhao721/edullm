#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=probe-olmo
#SBATCH --output=logs/probe_olmo_%j.out
#SBATCH --error=logs/probe_olmo_%j.err

set -euo pipefail
RUN_DIR=/scratch/users/nzhao2/agent-runs/refhq-unshard-20260727
cd "$RUN_DIR"
mkdir -p logs
source /scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/venv/bin/activate
which python3
python3 -c "import olmo_core; print('ok', olmo_core.__file__)"
