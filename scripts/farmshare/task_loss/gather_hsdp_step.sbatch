#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=gather-step
#SBATCH --output=logs/gather_%x_%j.out
#SBATCH --error=logs/gather_%x_%j.err

# Gather one legacy HSDP state.pt checkpoint to model_eval.pt (2 GPUs).
set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${CHECKPOINT_DIR:?}"
: "${OUT_PT:?}"

cd "${RUN_DIR}"
mkdir -p logs

# shellcheck disable=SC1090
source "${VENV}/bin/activate"

GATHER_PY="${GATHER_PY:-${RUN_DIR}/scripts/gather_hsdp_state_pt_to_model.py}"

if [[ -f "${OUT_PT}" ]]; then
  echo "SKIP gather (exists ${OUT_PT})"
  exit 0
fi

torchrun --standalone --nproc_per_node=2 \
  "${GATHER_PY}" \
  --checkpoint "${CHECKPOINT_DIR}" \
  --out "${OUT_PT}"

ls -lh "${OUT_PT}"
