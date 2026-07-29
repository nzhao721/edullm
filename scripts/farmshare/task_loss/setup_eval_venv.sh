#!/bin/bash
set -euo pipefail
RUN_DIR="${1:?}"
VENV="$RUN_DIR/venv"
LOG="$RUN_DIR/logs/venv_setup.out"
mkdir -p "$RUN_DIR/logs"

LADDER_PY=/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/venv/bin/python
"$LADDER_PY" - <<'PY'
from pathlib import Path
import zipfile
p = Path('/scratch/users/nzhao2/checkpoints/token-selection-370m/blade/checkpoints/step250/state.pt')
print('size', p.stat().st_size)
print('zip', zipfile.is_zipfile(p))
if zipfile.is_zipfile(p):
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        print('nentries', len(names))
        print('sample', names[:50])
PY

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip setuptools wheel

echo "pip torch start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision torchaudio
pip install "ai2-olmo" "ai2-olmo-core" transformers omegaconf rich cached_path \
  torchmetrics datasets boto3 requests
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import olmo, olmo_core, torchmetrics; print('olmo', olmo.__file__); print('olmo_core', olmo_core.__file__); print('torchmetrics ok')"
echo "pip torch done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
