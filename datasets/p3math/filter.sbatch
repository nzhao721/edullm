#!/usr/bin/env bash
# Filter PP2 subsets into P3Math outputs; copy Lean4 as-is.
#SBATCH --job-name=p3math-filter
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=wheat-01
#SBATCH --output=logs/filter_%j.out
#SBATCH --error=logs/filter_%j.err

set -euo pipefail
SCRATCH_ROOT="${SCRATCH_ROOT:?set SCRATCH_ROOT}"
REPO_ROOT="${REPO_ROOT:?set REPO_ROOT}"
cd "$SCRATCH_ROOT"
mkdir -p logs

export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/hf-cache}"
VENV="${VENV:-$SCRATCH_ROOT/venv}"
export PYTHONPATH="$REPO_ROOT/datasets${PYTHONPATH:+:$PYTHONPATH}"

"$VENV/bin/python" "$REPO_ROOT/datasets/p3math/filter_corpus.py" --scratch-root "$SCRATCH_ROOT" --skip-existing
