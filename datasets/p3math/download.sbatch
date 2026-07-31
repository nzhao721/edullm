#!/usr/bin/env bash
# Download PP2 + Lean4 + arXiv metadata to FarmShare scratch.
#SBATCH --job-name=p3math-dl
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=wheat-01
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

set -euo pipefail
SCRATCH_ROOT="${SCRATCH_ROOT:?set SCRATCH_ROOT}"
REPO_ROOT="${REPO_ROOT:?set REPO_ROOT}"
cd "$SCRATCH_ROOT"
mkdir -p logs

export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME"

VENV="${VENV:-$SCRATCH_ROOT/venv}"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -U pip
  "$VENV/bin/pip" install "huggingface_hub>=0.23" "datasets>=2.19" zstandard pyarrow
fi

export PYTHONPATH="$REPO_ROOT/datasets${PYTHONPATH:+:$PYTHONPATH}"
"$VENV/bin/python" "$REPO_ROOT/datasets/p3math/download_sources.py" --scratch-root "$SCRATCH_ROOT"
