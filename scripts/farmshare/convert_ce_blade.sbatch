#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=convert-370m
#SBATCH --output=logs/convert_%x_%j.out
#SBATCH --error=logs/convert_%x_%j.err

set -Eeuo pipefail
: "${RUN_DIR:?}"
: "${RUN_LABEL:?}"
: "${S3_CKPT_PREFIX:?}"

cd "$RUN_DIR"
mkdir -p logs
bash "$RUN_DIR/convert_state_pt_run.sh"
