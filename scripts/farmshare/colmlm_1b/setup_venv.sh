#!/usr/bin/env bash
# Setup venv for Co-LMLM 1B corpus assembly on FarmShare.
set -Eeuo pipefail
VENV="${VENV:-/scratch/users/nzhao2/agent-runs/venvs/colmlm-1b}"
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip wheel
pip install 'duckdb>=1.0' pyarrow 'huggingface_hub[hf_xet,cli]>=0.26' xxhash
python -c "import duckdb, pyarrow, huggingface_hub; print('deps ok', duckdb.__version__)"
