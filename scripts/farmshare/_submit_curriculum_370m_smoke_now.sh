#!/usr/bin/env bash
set -Eeuo pipefail

SOCK="${FARMSHARE_SOCK:-/tmp/farmshare-nzhao2.sock}"
HOST="${FARMSHARE_HOST:-nzhao2@login.farmshare.stanford.edu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="/scratch/users/nzhao2/agent-runs/curriculum-370m-smoke-${TS}"
LADDER_VENV="/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/venv"
REGMIX_TOKENIZED="/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810/tokenized"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "mkdir -p '${RUN_DIR}/logs' '${RUN_DIR}/ckpts' '${RUN_DIR}/progress' '${RUN_DIR}/cache'"

tar -C "${ROOT}" -czf - \
  experiments/curriculum/train_curriculum_regmix_370m.py \
  experiments/curriculum/curriculum_pacing.py \
  experiments/curriculum/launch/launch_arm.sh \
  experiments/token-selection/token_selection |
  ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "tar -xzf - -C '${RUN_DIR}'"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "find '${REGMIX_TOKENIZED}' -name '*.npy' | sort > '${RUN_DIR}/paths_train.txt'"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "cat > '${RUN_DIR}/run_smoke.sh'" <<'EOS'
#!/usr/bin/env bash
set -Eeuo pipefail
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load python/3.12.3 2>/dev/null || true
module load cuda/12.9.0 2>/dev/null || module load cuda/12.4.0 2>/dev/null || true

export OLMO_FLASH_ATTENTION=0
export OLMO_ATTN_BACKEND=torch
export OLMO_FUSED_LOSS=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# shellcheck disable=SC1091
source "${LADDER_VENV}/bin/activate"

export PYTHONPATH="${RUN_DIR}/experiments/curriculum:${RUN_DIR}/experiments/token-selection:${PYTHONPATH:-}"

nvidia-smi -L
"${LADDER_VENV}/bin/python" -c "import olmo_core, torch; print('olmo_core ok', 'cuda', torch.cuda.is_available())"

ARM_ID=control-smoke \
PACING=control \
SAVE_FOLDER="${RUN_DIR}/ckpts" \
PROGRESS_DIR="${RUN_DIR}/progress" \
DATA_CACHE_DIR="${RUN_DIR}/cache" \
TRAIN_PATHS_FILE="${RUN_DIR}/paths_train.txt" \
RUN_DIR="${RUN_DIR}" \
NPROC=1 \
DEVICE_BATCH_SIZE=8 \
FRESH=1 \
WANDB_MODE=disabled \
S3_EXPORT=0 \
ALLOW_LOCAL_ONLY=1 \
EXTRA_ARGS="--length-tokens 83886080 --log-interval 2 --num-workers 0 --no-task-loss-on-save --no-s3-export --wandb-mode disabled --allow-local-only" \
PYTHON="${LADDER_VENV}/bin/python" \
TRAIN_SCRIPT="${RUN_DIR}/experiments/curriculum/train_curriculum_regmix_370m.py" \
bash "${RUN_DIR}/experiments/curriculum/launch/launch_arm.sh"
EOS

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "sed -i 's/\r$//' '${RUN_DIR}/run_smoke.sh' '${RUN_DIR}/experiments/curriculum/launch/launch_arm.sh' && chmod +x '${RUN_DIR}/run_smoke.sh' '${RUN_DIR}/experiments/curriculum/launch/launch_arm.sh'"

JOB_ID="$(ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "sbatch --parsable --exclude=wheat-01 \
    --partition=gpu \
    --qos=gpu \
    --gpus-per-node=1 \
    --cpus-per-task=8 \
    --mem=128G \
    --time=01:00:00 \
    --job-name=cur370m-smoke \
    --chdir='${RUN_DIR}' \
    --output='${RUN_DIR}/logs/train-%j.out' \
    --error='${RUN_DIR}/logs/train-%j.err' \
    --export=ALL,RUN_DIR='${RUN_DIR}',LADDER_VENV='${LADDER_VENV}' \
    '${RUN_DIR}/run_smoke.sh'")"

echo "job_id=${JOB_ID}"
echo "RUN_DIR=${RUN_DIR}"
echo "submitted curriculum-370m-smoke=${JOB_ID}"
