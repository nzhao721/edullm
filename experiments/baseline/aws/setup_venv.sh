#!/usr/bin/env bash
# Create (or reuse) a Python venv with everything needed for the OLMo-ladder
# 370M AWS pipeline on a p5.48xlarge (8x H100). AWS-only: no W&B, no
# FarmShare/Slurm modules, no Colab.
set -euo pipefail

WORK="${WORK:-/opt/edullm/run}"
VENV_DIR="${VENV_DIR:-$WORK/venv}"

log() { echo "[$(date -Is)] $*"; }

export DEBIAN_FRONTEND=noninteractive
mkdir -p "$WORK"

# ---------------------------------------------------------------------------
# System deps (best-effort; Deep Learning AMIs already ship python3/git/CUDA).
# ---------------------------------------------------------------------------
if ! command -v git >/dev/null 2>&1 || { ! command -v python3.11 >/dev/null 2>&1 && ! command -v python3.10 >/dev/null 2>&1 && ! python3 -m venv --help >/dev/null 2>&1; }; then
  log "installing base packages (python3-venv, git) via apt"
  (sudo -n true 2>/dev/null && SUDO=sudo) || SUDO=""
  $SUDO apt-get update -y || true
  $SUDO apt-get install -y python3-venv python3-pip git || true
fi

PYBIN="$(command -v python3.11 || command -v python3.10 || command -v python3)"
log "using python: $PYBIN ($($PYBIN --version 2>&1))"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  log "creating venv at $VENV_DIR"
  "$PYBIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel setuptools

# ---------------------------------------------------------------------------
# Torch: reuse the AMI-provided CUDA build if present (common on AWS Deep
# Learning AMIs / PyTorch DLAMIs), otherwise install torch 2.4.1+cu124.
# ---------------------------------------------------------------------------
if "$VENV_DIR/bin/python" -c "import torch" >/dev/null 2>&1; then
  log "torch already importable in venv: $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
elif [[ -n "${AMI_TORCH_SITE_PACKAGES:-}" && -d "${AMI_TORCH_SITE_PACKAGES}" ]]; then
  log "linking AMI-provided torch from ${AMI_TORCH_SITE_PACKAGES}"
  SITE_DIR="$(python -c 'import site; print(site.getsitepackages()[0])')"
  echo "${AMI_TORCH_SITE_PACKAGES}" > "${SITE_DIR}/ami_torch.pth"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
else
  log "installing torch==2.4.1+cu124"
  python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
fi

# ---------------------------------------------------------------------------
# Training + tokenization deps
# ---------------------------------------------------------------------------
log "installing ai2-olmo and training/tokenization dependencies"
python -m pip install --no-deps 'ai2-olmo==0.6.0' || python -m pip install 'ai2-olmo==0.6.0'
python -m pip install \
  'numpy<2.1' \
  'transformers>=4.44' \
  'huggingface_hub' \
  'zstandard' \
  'tqdm' \
  'sentencepiece' \
  'protobuf' \
  'omegaconf' \
  'rich' \
  'safetensors' \
  'cached-path' \
  'packaging' \
  'requests' \
  'importlib_resources' \
  'boto3' \
  'einops'

# Optional flash-attn build (H100/sm90 supports it; ai2-olmo falls back to
# eager attention if this is unavailable, controlled via OLMO_FLASH_ATTENTION).
python -m pip install ninja || true
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
if python -m pip install flash-attn==2.5.9.post1 --no-build-isolation; then
  log "flash-attn installed"
else
  log "flash-attn build failed; falling back to OLMO_FLASH_ATTENTION=0"
  grep -q '^export OLMO_FLASH_ATTENTION=' "$WORK/env.sh" 2>/dev/null || echo 'export OLMO_FLASH_ATTENTION=0' >> "$WORK/env.sh"
fi
# On H100 we want flash attention on by default unless the block above disabled it.
grep -q '^export OLMO_FLASH_ATTENTION=' "$WORK/env.sh" 2>/dev/null || echo 'export OLMO_FLASH_ATTENTION=1' >> "$WORK/env.sh"

# ---------------------------------------------------------------------------
# OLMo-ladder repo (training entrypoint: src/ladder/train.py)
# ---------------------------------------------------------------------------
if [[ ! -d "$WORK/OLMo-ladder/.git" ]]; then
  log "cloning allenai/OLMo-ladder"
  git clone --depth 1 https://github.com/allenai/OLMo-ladder.git "$WORK/OLMo-ladder"
fi
grep -q '^export OLMO_LADDER_ROOT=' "$WORK/env.sh" 2>/dev/null || echo "export OLMO_LADDER_ROOT=\"$WORK/OLMo-ladder\"" >> "$WORK/env.sh"

# Explicitly isolate from W&B, regardless of ambient environment.
grep -q '^export WANDB_DISABLED=' "$WORK/env.sh" 2>/dev/null || echo 'export WANDB_DISABLED=1' >> "$WORK/env.sh"
grep -q '^export WANDB_MODE=' "$WORK/env.sh" 2>/dev/null || echo 'export WANDB_MODE=disabled' >> "$WORK/env.sh"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu_count", torch.cuda.device_count())
import olmo  # noqa: F401
print("ai2-olmo import OK")
PY

log "venv ready at $VENV_DIR"
