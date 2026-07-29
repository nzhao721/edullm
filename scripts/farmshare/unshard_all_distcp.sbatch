#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=unshard-all
#SBATCH --output=logs/unshard_all_%x_%j.out
#SBATCH --error=logs/unshard_all_%x_%j.err

set -Eeuo pipefail
: "${RUN_DIR:?}"
: "${RUN_LABEL:?}"

cd "$RUN_DIR"
mkdir -p logs
bash "$RUN_DIR/unshard_olmo_core_run.sh"
