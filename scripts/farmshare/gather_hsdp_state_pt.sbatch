#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=gather-hsdp
#SBATCH --output=logs/gather_hsdp_%j.out
#SBATCH --error=logs/gather_hsdp_%j.err

set -Eeuo pipefail
: "${RUN_DIR:?}"
: "${CHECKPOINT_DIR:?}"
: "${OUT_PT:?}"

cd "$RUN_DIR"
mkdir -p logs

LADDER_VENV="${LADDER_VENV:-/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/venv}"
# shellcheck disable=SC1091
source "$LADDER_VENV/bin/activate"

GATHER_PY="${GATHER_PY:-$RUN_DIR/gather_hsdp_state_pt_to_model.py}"

torchrun --standalone --nproc_per_node=2 \
  "$GATHER_PY" \
  --checkpoint "$CHECKPOINT_DIR" \
  --out "$OUT_PT"

ls -lh "$OUT_PT"
