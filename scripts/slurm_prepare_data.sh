#!/bin/bash
#SBATCH --job-name=edullm-prep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/users/nzhao2/edullm
#SBATCH --output=logs/prepare_%x_%j.out
#SBATCH --error=logs/prepare_%x_%j.err

set -eo pipefail

echo "=== prepare_data job start ==="
echo "DATASET=${DATASET:-<unset>} MAX_TOKENS=${MAX_TOKENS:-<unset>} OUT_DIR=${OUT_DIR:-<unset>}"

# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-/home/users/nzhao2/edullm}/scripts/slurm_common.sh"
slurm_project_setup

: "${DATASET:?Set DATASET=fineweb_edu or slimpajama}"
: "${MAX_TOKENS:?Set MAX_TOKENS}"
: "${OUT_DIR:?Set OUT_DIR, e.g. data/slimpajama}"

python -u scripts/prepare_data.py \
  --dataset "${DATASET}" \
  --output-dir "${OUT_DIR}" \
  --max-train-tokens "${MAX_TOKENS}" \
  --val-tokens-target 320000

echo "=== prepare_data job done ==="
