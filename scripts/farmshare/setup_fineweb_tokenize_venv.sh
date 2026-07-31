#!/usr/bin/env bash
set -Eeuo pipefail

VENV="${VENV:-/scratch/users/nzhao2/agent-runs/venvs/fineweb-tokenize}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip wheel
pip install datasets transformers numpy tqdm huggingface_hub
python -c "import datasets, transformers; print('deps ok')"
