#!/usr/bin/env bash
set -Eeuo pipefail

VENV="${VENV:-/scratch/users/nzhao2/agent-runs/venvs/smollm2-train}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip wheel setuptools

pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision torchaudio
pip install "transformers>=4.44" numpy tqdm huggingface_hub safetensors datasets

python - <<'PY'
import torch
from transformers import AutoConfig, AutoModelForCausalLM

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
cfg = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_config(cfg)
print("smollm2 params", sum(p.numel() for p in model.parameters()))
PY
