#!/usr/bin/env bash
# Bootstrap a SmolLM2 training venv (torch + transformers + edullm-data + wandb).
# Prefer a job-scoped path: VENV=$RUN_DIR/venv (ephemeral empty-scratch friendly).
set -Eeuo pipefail

SUNET="${SUNET:-${USER:-nzhao2}}"
VENV="${VENV:-/scratch/users/${SUNET}/agent-runs/venvs/smollm2-train}"

mkdir -p "$(dirname "${VENV}")"

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip wheel setuptools

pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision torchaudio
pip install "transformers>=4.44" numpy tqdm huggingface_hub safetensors datasets \
  boto3 "wandb>=0.17" \
  "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0"

python - <<'PY'
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import edullm_data
from edullm_data.read import dataset_paths, resolve_latest

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("edullm_data", getattr(edullm_data, "__version__", "ok"))
print("read_api", dataset_paths.__name__, resolve_latest.__name__)
cfg = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_config(cfg)
print("smollm2 params", sum(p.numel() for p in model.parameters()))
PY

echo "venv ready: ${VENV}"
