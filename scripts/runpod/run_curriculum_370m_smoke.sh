#!/usr/bin/env bash
# RunPod L40S smoke: OLMo2-370M curriculum control arm (~20 steps).
set -Eeuo pipefail

RUN_DIR="${RUN_DIR:-/workspace/curriculum-370m-smoke}"
REPO_DIR="${REPO_DIR:-/workspace/edullm}"
AWS_ENV="${AWS_ENV:-/workspace/aws-session.env}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"

export OLMO_FLASH_ATTENTION=1
export OLMO_ATTN_BACKEND=flash_2
export OLMO_FUSED_LOSS=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p "${RUN_DIR}"/{logs,ckpts,progress,cache,data}

if [[ -f "${AWS_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${AWS_ENV}"
fi

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/nzhao721/edullm.git "${REPO_DIR}"
fi

PYTHON=python3
# Keep the image's CUDA-matched torch; pin it while installing olmo-core deps.
TORCH_VER="$(${PYTHON} -c 'import torch; print(torch.__version__)')"
echo "torch==${TORCH_VER}" > /tmp/pip-constraints.txt
${PYTHON} -m pip install -q -U pip wheel
${PYTHON} -m pip install -q -c /tmp/pip-constraints.txt \
  "ai2-olmo-core>=2.0" \
  "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0" \
  boto3 wandb pyyaml tqdm
${PYTHON} -m pip install -q --no-cache-dir \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
${PYTHON} -c "import flash_attn; print('flash_attn ok', flash_attn.__version__)"

# Minimal train shard (~600MB) — enough for 84M-token smoke.
DATA_DIR="${RUN_DIR}/data/regmix-wiki"
export DATA_DIR
mkdir -p "${DATA_DIR}"
${PYTHON} - <<'PY'
import os
from pathlib import Path
import boto3

dest = Path(os.environ["DATA_DIR"]) / "train-00000.u32le.bin"
dest.parent.mkdir(parents=True, exist_ok=True)
bucket, key = "edullm-data", "pretrain/regmix-10b/v1/tokens/wiki/train-00000.u32le.bin"
boto3.client("s3").download_file(bucket, key, str(dest))
print("downloaded", dest, dest.stat().st_size)
PY
printf '%s\n' "${DATA_DIR}/train-00000.u32le.bin" > "${RUN_DIR}/paths_train.txt"

export PYTHONPATH="${REPO_DIR}/experiments/curriculum:${REPO_DIR}/experiments/token-selection:${PYTHONPATH:-}"

nvidia-smi -L
for i in $(seq 1 36); do
  if ${PYTHON} -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    break
  fi
  echo "waiting for cuda ($i/36)..."
  sleep 5
done
${PYTHON} -c "import olmo_core, torch; print('olmo_core ok', 'cuda', torch.cuda.is_available())"

export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 NUM_NODES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500

ARM_ID=control-smoke \
PACING=control \
SAVE_FOLDER="${RUN_DIR}/ckpts" \
PROGRESS_DIR="${RUN_DIR}/progress" \
DATA_CACHE_DIR="${RUN_DIR}/cache" \
TRAIN_PATHS_FILE="${RUN_DIR}/paths_train.txt" \
RUN_DIR="${RUN_DIR}" \
NPROC=1 \
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE}" \
FRESH=1 \
WANDB_MODE=disabled \
S3_EXPORT=0 \
ALLOW_LOCAL_ONLY=1 \
EXTRA_ARGS="--length-tokens 12582912 --log-interval 1 --num-workers 0 --no-task-loss-on-save --no-s3-export --wandb-mode disabled --allow-local-only" \
PYTHON="${PYTHON}" \
TRAIN_SCRIPT="${REPO_DIR}/experiments/curriculum/train_curriculum_regmix_370m.py" \
bash "${REPO_DIR}/experiments/curriculum/launch/launch_arm.sh" \
  2>&1 | tee "${RUN_DIR}/logs/smoke.log"

echo "SMOKE_DONE log=${RUN_DIR}/logs/smoke.log"
tail -n 40 "${RUN_DIR}/logs/smoke.log" || true
# Keep pod alive so logs can be fetched.
sleep 7200
