#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=olmo-label
#SBATCH --output=logs/label-%A_%a.out
#SBATCH --error=logs/label-%A_%a.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${MANIFEST:?}" "${LABELS_ROOT:?}"

source "${VENV}/bin/activate"
export PYTHONPATH="${RUN_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${LABELS_ROOT}" "${RUN_DIR}/logs"

WORKERS="${SLURM_CPUS_PER_TASK:-16}"
# Leave one core for the parent process / I/O.
if [[ "${WORKERS}" -gt 1 ]]; then
  WORKERS=$((WORKERS - 1))
fi

python "${RUN_DIR}/scripts/label_olmo_shard.py" \
  --manifest "${MANIFEST}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --out-root "${LABELS_ROOT}" \
  --workers "${WORKERS}" \
  --batch-size 64
