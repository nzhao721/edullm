#!/usr/bin/env bash
# 4-GPU SmolLM2-135M pretrain on 500M-token FineWeb-Edu subset, 40 epochs (20B tokens).
# Default layout: 2 nodes x 2 GPUs (multi-node DDP).
#
# Usage (FarmShare login node):
#   bash scripts/farmshare/submit_smollm2_135m_500m_40ep.sh
#
# Single-node 4-GPU (previous behavior):
#   NUM_NODES=1 GPUS_PER_NODE=4 bash scripts/farmshare/submit_smollm2_135m_500m_40ep.sh
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-smollm2-135m-500m-40ep-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
SRC_DATA_DIR="${SRC_DATA_DIR:-/scratch/users/${SUNET}/agent-runs/fineweb-edu-1b-smollm2-tokenized}"
DATA_DIR="${DATA_DIR:-/scratch/users/${SUNET}/agent-runs/fineweb-edu-500m-smollm2-tokenized}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/output}"
VENV="${VENV:-/scratch/users/${SUNET}/agent-runs/venvs/smollm2-train}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLICE_PY="${SLICE_PY:-${SCRIPT_DIR}/slice_tokenized_subset.py}"
TRAIN_PY="${TRAIN_PY:-${SCRIPT_DIR}/train_smollm2_135m_ddp.py}"
HF_HOME="${HF_HOME:-/scratch/users/${SUNET}/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"

NUM_NODES="${NUM_NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
NUM_GPUS="${NUM_GPUS:-$((NUM_NODES * GPUS_PER_NODE))}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-20000000000}"
SUBSET_TOKENS="${SUBSET_TOKENS:-500000000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_EVERY="${LOG_EVERY:-20}"
MASTER_PORT="${MASTER_PORT:-29500}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
SLURM_MEM="${SLURM_MEM:-128G}"
RESUME_FROM="${RESUME_FROM:-}"
NODELIST="${NODELIST:-}"

mkdir -p "${RUN_DIR}/logs" "${OUT_DIR}" "${HF_HOME}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "missing venv python at ${VENV}/bin/python" >&2
  exit 2
fi
for f in "${SLICE_PY}" "${TRAIN_PY}"; do
  if [[ ! -f "${f}" ]]; then
    echo "missing ${f}" >&2
    exit 2
  fi
done

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
OUT_DIR=${OUT_DIR}
SRC_DATA_DIR=${SRC_DATA_DIR}
DATA_DIR=${DATA_DIR}
VENV=${VENV}
SLICE_PY=${SLICE_PY}
TRAIN_PY=${TRAIN_PY}
HF_HOME=${HF_HOME}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}
NUM_NODES=${NUM_NODES}
GPUS_PER_NODE=${GPUS_PER_NODE}
NUM_GPUS=${NUM_GPUS}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}
NUM_EPOCHS=${NUM_EPOCHS}
MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS}
SUBSET_TOKENS=${SUBSET_TOKENS}
NUM_WORKERS=${NUM_WORKERS}
LOG_EVERY=${LOG_EVERY}
MASTER_PORT=${MASTER_PORT}
CPUS_PER_TASK=${CPUS_PER_TASK}
SLURM_MEM=${SLURM_MEM}
RESUME_FROM=${RESUME_FROM}
NODELIST=${NODELIST}
EOF

cat > "${RUN_DIR}/launch_train.sh" <<'LAUNCH_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load python/3.12.3 2>/dev/null || true
module load cuda/12.9.0 2>/dev/null || module load cuda/12.4.0 2>/dev/null || true
source "${RUN_DIR}/env.sh"
source "${VENV}/bin/activate"

export HF_HOME TRANSFORMERS_CACHE TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

test -f "${DATA_DIR}/meta.json" || { echo "missing ${DATA_DIR}/meta.json" >&2; exit 2; }

echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_NNODES=${SLURM_NNODES:-}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
nvidia-smi -L || true

TRAIN_ARGS=(
  "${TRAIN_PY}"
  --data-dir "${DATA_DIR}"
  --output-dir "${OUT_DIR}"
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
  --num-epochs "${NUM_EPOCHS}"
  --max-train-tokens "${MAX_TRAIN_TOKENS}"
  --checkpoint-interval-epochs 0.5
  --eval-interval-epochs 0.5
  --num-workers "${NUM_WORKERS}"
  --log-every "${LOG_EVERY}"
)
if [[ -n "${RESUME_FROM}" ]]; then
  TRAIN_ARGS+=(--resume-from "${RESUME_FROM}")
fi

if [[ "${NUM_NODES}" -eq 1 ]]; then
  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${GPUS_PER_NODE}" \
    "${TRAIN_ARGS[@]}"
else
  MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
  export MASTER_ADDR
  echo "MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
  srun --ntasks-per-node=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --gpus-per-node="${GPUS_PER_NODE}" \
    bash -c "
    set -Eeuo pipefail
    source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
    module load python/3.12.3 2>/dev/null || true
    module load cuda/12.9.0 2>/dev/null || module load cuda/12.4.0 2>/dev/null || true
    source \"${RUN_DIR}/env.sh\"
    source \"\${VENV}/bin/activate\"
    export HF_HOME TRANSFORMERS_CACHE TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_DEBUG=WARN
    RESUME_ARGS=
    if [[ -n \"${RESUME_FROM}\" ]]; then
      RESUME_ARGS=\"--resume-from ${RESUME_FROM}\"
    fi
    python -m torch.distributed.run \
      --nnodes=\${SLURM_NNODES} \
      --nproc_per_node=${GPUS_PER_NODE} \
      --rdzv_id=\${SLURM_JOB_ID} \
      --rdzv_backend=c10d \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      ${TRAIN_PY} \
      --data-dir ${DATA_DIR} \
      --output-dir ${OUT_DIR} \
      --per-device-batch-size ${PER_DEVICE_BATCH_SIZE} \
      --num-epochs ${NUM_EPOCHS} \
      --max-train-tokens ${MAX_TRAIN_TOKENS} \
      --checkpoint-interval-epochs 0.5 \
      --eval-interval-epochs 0.5 \
      --num-workers ${NUM_WORKERS} \
      --log-every ${LOG_EVERY} \
      \${RESUME_ARGS}
  "
fi
LAUNCH_EOF
chmod +x "${RUN_DIR}/launch_train.sh"

cat > "${RUN_DIR}/train.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=smollm2-40ep
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=12:00:00
#SBATCH --exclude=wheat-01
$(if [[ -n "${NODELIST}" ]]; then echo "#SBATCH --nodelist=${NODELIST}"; fi)
#SBATCH --chdir=${RUN_DIR}
#SBATCH --output=${RUN_DIR}/logs/train-%j.out
#SBATCH --error=${RUN_DIR}/logs/train-%j.er

export RUN_DIR=${RUN_DIR}
bash "${RUN_DIR}/launch_train.sh"
EOF

JOB_ID=$(sbatch --parsable "${RUN_DIR}/train.sbatch")

echo "job_id=${JOB_ID}"
echo "${JOB_ID}" > "${RUN_DIR}/job_id.txt"
echo "RUN_DIR=${RUN_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "NUM_NODES=${NUM_NODES}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "SLURM_MEM=${SLURM_MEM}"
echo "RESUME_FROM=${RESUME_FROM}"
echo "NODELIST=${NODELIST}"
echo "NUM_EPOCHS=${NUM_EPOCHS}"
echo "MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS}"
echo "submitted smollm2-40ep=${JOB_ID}"
