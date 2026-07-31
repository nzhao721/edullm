#!/usr/bin/env bash
# 4-GPU SmolLM2-135M pretrain on published FineWeb-Edu 500M tokens, 40 epochs (20B tokens).
# Default layout: 2 nodes x 2 GPUs (multi-node DDP).
#
# Ephemeral scratch contract:
#   - Data: resolve s3://edullm-data/<DATASET_ID>/<version>/ and stage into
#     ${RUN_DIR}/staged-data (job-scoped; no shared edullm-cache / FineWeb slice).
#   - Durable artifacts: upload checkpoints/evals to CHECKPOINT_S3_URI and/or W&B online.
#   - Does not assume a pre-existing tokenized corpus, local ckpt, or s3://edullm-datasets/.
#
# Prerequisites:
#   - venv with torch + edullm-data + boto3 (setup_smollm2_train_venv.sh; sync creates it)
#   - ${RUN_DIR}/wandb-session.env with WANDB_API_KEY (unless relying solely on S3)
#   - ${RUN_DIR}/aws-session.env with temporary AWS creds that can read edullm-data
#     and write edullm-checkpoints (mint on laptop; FarmShare cannot use sb-aws-creds login)
#
# Usage (FarmShare login node):
#   bash scripts/farmshare/submit_smollm2_135m_500m_40ep.sh
#
# Single-node 4-GPU:
#   NUM_NODES=1 GPUS_PER_NODE=4 bash scripts/farmshare/submit_smollm2_135m_500m_40ep.sh
set -Eeuo pipefail

SUNET="${SUNET:-${USER:-nzhao2}}"
RUN_NAME="${RUN_NAME:-smollm2-135m-500m-40ep-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
DATASET_ID="${DATASET_ID:-pretrain/fineweb-edu-500m}"
DATASET_VERSION="${DATASET_VERSION:-}"
SPLIT="${SPLIT:-train}"
STAGE_DIR="${STAGE_DIR:-${RUN_DIR}/staged-data}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/output}"
VENV="${VENV:-/scratch/users/${SUNET}/agent-runs/venvs/smollm2-train}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${TRAIN_PY:-${SCRIPT_DIR}/train_smollm2_135m_ddp.py}"
HF_HOME="${HF_HOME:-${RUN_DIR}/hf-cache}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
CHECKPOINT_S3_URI="${CHECKPOINT_S3_URI:-s3://edullm-checkpoints/smollm2/${RUN_NAME}}"
RESUME_FROM_S3="${RESUME_FROM_S3:-}"

NUM_NODES="${NUM_NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
NUM_GPUS="${NUM_GPUS:-$((NUM_NODES * GPUS_PER_NODE))}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-20000000000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_EVERY="${LOG_EVERY:-20}"
MASTER_PORT="${MASTER_PORT:-29500}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
SLURM_MEM="${SLURM_MEM:-128G}"
RESUME_FROM="${RESUME_FROM:-}"
NODELIST="${NODELIST:-}"
WANDB_PROJECT="${WANDB_PROJECT:-edullm-smollm2}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_UPLOAD_EXISTING="${WANDB_UPLOAD_EXISTING:-0}"
CHECKPOINT_INTERVAL_TOKENS="${CHECKPOINT_INTERVAL_TOKENS:-250000000}"
EVAL_INTERVAL_TOKENS="${EVAL_INTERVAL_TOKENS:-250000000}"

mkdir -p "${RUN_DIR}/logs" "${OUT_DIR}" "${HF_HOME}" "${STAGE_DIR}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "missing venv python at ${VENV}/bin/python" >&2
  echo "Run: SUNET=${SUNET} bash ${SCRIPT_DIR}/setup_smollm2_train_venv.sh" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "missing ${TRAIN_PY}" >&2
  exit 2
fi
if [[ ! -f "${RUN_DIR}/aws-session.env" ]]; then
  echo "missing ${RUN_DIR}/aws-session.env (mint AWS session on laptop and push to RUN_DIR)" >&2
  exit 2
fi
if [[ "${WANDB_MODE}" == "online" && ! -f "${RUN_DIR}/wandb-session.env" ]]; then
  echo "missing ${RUN_DIR}/wandb-session.env (required for WANDB_MODE=online durable upload)" >&2
  exit 2
fi
if [[ -z "${CHECKPOINT_S3_URI}" && "${WANDB_MODE}" != "online" ]]; then
  echo "durable save required: set CHECKPOINT_S3_URI and/or WANDB_MODE=online" >&2
  exit 2
fi
if [[ -n "${RESUME_FROM}" ]]; then
  echo "WARNING: RESUME_FROM points at local scratch; prefer RESUME_FROM_S3 for ephemeral runs" >&2
fi

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
OUT_DIR=${OUT_DIR}
DATASET_ID=${DATASET_ID}
DATASET_VERSION=${DATASET_VERSION}
SPLIT=${SPLIT}
STAGE_DIR=${STAGE_DIR}
VENV=${VENV}
TRAIN_PY=${TRAIN_PY}
HF_HOME=${HF_HOME}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}
CHECKPOINT_S3_URI=${CHECKPOINT_S3_URI}
RESUME_FROM_S3=${RESUME_FROM_S3}
NUM_NODES=${NUM_NODES}
GPUS_PER_NODE=${GPUS_PER_NODE}
NUM_GPUS=${NUM_GPUS}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}
NUM_EPOCHS=${NUM_EPOCHS}
MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS}
NUM_WORKERS=${NUM_WORKERS}
LOG_EVERY=${LOG_EVERY}
MASTER_PORT=${MASTER_PORT}
CPUS_PER_TASK=${CPUS_PER_TASK}
SLURM_MEM=${SLURM_MEM}
RESUME_FROM=${RESUME_FROM}
NODELIST=${NODELIST}
WANDB_PROJECT=${WANDB_PROJECT}
WANDB_ENTITY=${WANDB_ENTITY}
WANDB_RUN_NAME=${WANDB_RUN_NAME}
WANDB_MODE=${WANDB_MODE}
WANDB_UPLOAD_EXISTING=${WANDB_UPLOAD_EXISTING}
CHECKPOINT_INTERVAL_TOKENS=${CHECKPOINT_INTERVAL_TOKENS}
EVAL_INTERVAL_TOKENS=${EVAL_INTERVAL_TOKENS}
EOF

cat > "${RUN_DIR}/launch_train.sh" <<'LAUNCH_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load python/3.12.3 2>/dev/null || true
module load cuda/12.9.0 2>/dev/null || module load cuda/12.4.0 2>/dev/null || true
source "${RUN_DIR}/env.sh"
# shellcheck disable=SC1091
if [[ -f "${RUN_DIR}/wandb-session.env" ]]; then
  source "${RUN_DIR}/wandb-session.env"
fi
# shellcheck disable=SC1091
source "${RUN_DIR}/aws-session.env"
source "${VENV}/bin/activate"

export HF_HOME TRANSFORMERS_CACHE TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export WANDB_PROJECT WANDB_ENTITY WANDB_RUN_NAME WANDB_MODE
export WANDB_DIR="${OUT_DIR}/wandb"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export CHECKPOINT_S3_URI
mkdir -p "${WANDB_DIR}" "${STAGE_DIR}"

test -n "${AWS_ACCESS_KEY_ID:-}" || { echo "AWS_ACCESS_KEY_ID missing after sourcing aws-session.env" >&2; exit 2; }
test -n "${AWS_SECRET_ACCESS_KEY:-}" || { echo "AWS_SECRET_ACCESS_KEY missing after sourcing aws-session.env" >&2; exit 2; }
if [[ "${WANDB_MODE}" == "online" ]]; then
  test -n "${WANDB_API_KEY:-}" || { echo "WANDB_API_KEY missing after sourcing wandb-session.env" >&2; exit 2; }
fi

echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_NNODES=${SLURM_NNODES:-}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
echo "DATASET_ID=${DATASET_ID} DATASET_VERSION=${DATASET_VERSION:-latest} STAGE_DIR=${STAGE_DIR}"
echo "CHECKPOINT_S3_URI=${CHECKPOINT_S3_URI}"
echo "WANDB_PROJECT=${WANDB_PROJECT} WANDB_MODE=${WANDB_MODE}"
nvidia-smi -L || true

TRAIN_ARGS=(
  "${TRAIN_PY}"
  --dataset-id "${DATASET_ID}"
  --split "${SPLIT}"
  --stage-dir "${STAGE_DIR}"
  --output-dir "${OUT_DIR}"
  --run-name "${WANDB_RUN_NAME}"
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}"
  --num-epochs "${NUM_EPOCHS}"
  --max-train-tokens "${MAX_TRAIN_TOKENS}"
  --checkpoint-interval-epochs 0.5
  --eval-interval-epochs 0.5
  --num-workers "${NUM_WORKERS}"
  --log-every "${LOG_EVERY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode "${WANDB_MODE}"
  --wandb-run-name "${WANDB_RUN_NAME}"
  --checkpoint-interval-tokens "${CHECKPOINT_INTERVAL_TOKENS}"
  --eval-interval-tokens "${EVAL_INTERVAL_TOKENS}"
)
if [[ -n "${CHECKPOINT_S3_URI}" ]]; then
  TRAIN_ARGS+=(--checkpoint-s3-uri "${CHECKPOINT_S3_URI}")
fi
if [[ -n "${DATASET_VERSION}" ]]; then
  TRAIN_ARGS+=(--dataset-version "${DATASET_VERSION}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  TRAIN_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ "${WANDB_UPLOAD_EXISTING}" == "1" ]]; then
  TRAIN_ARGS+=(--wandb-upload-existing)
fi
if [[ -n "${RESUME_FROM_S3}" ]]; then
  TRAIN_ARGS+=(--resume-from-s3 "${RESUME_FROM_S3}")
elif [[ -n "${RESUME_FROM}" ]]; then
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
  VERSION_ARG=""
  if [[ -n "${DATASET_VERSION}" ]]; then
    VERSION_ARG="--dataset-version ${DATASET_VERSION}"
  fi
  WANDB_ENTITY_ARG=""
  if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_ENTITY_ARG="--wandb-entity ${WANDB_ENTITY}"
  fi
  WANDB_EXISTING_ARG=""
  if [[ "${WANDB_UPLOAD_EXISTING}" == "1" ]]; then
    WANDB_EXISTING_ARG="--wandb-upload-existing"
  fi
  CKPT_S3_ARG=""
  if [[ -n "${CHECKPOINT_S3_URI}" ]]; then
    CKPT_S3_ARG="--checkpoint-s3-uri ${CHECKPOINT_S3_URI}"
  fi
  RESUME_ARGS=""
  if [[ -n "${RESUME_FROM_S3}" ]]; then
    RESUME_ARGS="--resume-from-s3 ${RESUME_FROM_S3}"
  elif [[ -n "${RESUME_FROM}" ]]; then
    RESUME_ARGS="--resume-from ${RESUME_FROM}"
  fi
  srun --ntasks-per-node=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --gpus-per-node="${GPUS_PER_NODE}" \
    bash -c "
    set -Eeuo pipefail
    source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
    module load python/3.12.3 2>/dev/null || true
    module load cuda/12.9.0 2>/dev/null || module load cuda/12.4.0 2>/dev/null || true
    source \"${RUN_DIR}/env.sh\"
    if [[ -f \"${RUN_DIR}/wandb-session.env\" ]]; then source \"${RUN_DIR}/wandb-session.env\"; fi
    source \"${RUN_DIR}/aws-session.env\"
    source \"\${VENV}/bin/activate\"
    export HF_HOME TRANSFORMERS_CACHE TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_DEBUG=WARN
    export WANDB_PROJECT WANDB_ENTITY WANDB_RUN_NAME WANDB_MODE
    export WANDB_DIR=\"${OUT_DIR}/wandb\"
    export AWS_DEFAULT_REGION=\"\${AWS_DEFAULT_REGION:-us-east-1}\"
    export CHECKPOINT_S3_URI
    mkdir -p \"\${WANDB_DIR}\" \"\${STAGE_DIR}\"
    python -m torch.distributed.run \
      --nnodes=\${SLURM_NNODES} \
      --nproc_per_node=${GPUS_PER_NODE} \
      --rdzv_id=\${SLURM_JOB_ID} \
      --rdzv_backend=c10d \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      ${TRAIN_PY} \
      --dataset-id ${DATASET_ID} \
      --split ${SPLIT} \
      --stage-dir ${STAGE_DIR} \
      --output-dir ${OUT_DIR} \
      --run-name ${WANDB_RUN_NAME} \
      --per-device-batch-size ${PER_DEVICE_BATCH_SIZE} \
      --num-epochs ${NUM_EPOCHS} \
      --max-train-tokens ${MAX_TRAIN_TOKENS} \
      --checkpoint-interval-epochs 0.5 \
      --eval-interval-epochs 0.5 \
      --num-workers ${NUM_WORKERS} \
      --log-every ${LOG_EVERY} \
      --wandb-project ${WANDB_PROJECT} \
      --wandb-mode ${WANDB_MODE} \
      --wandb-run-name ${WANDB_RUN_NAME} \
      --checkpoint-interval-tokens ${CHECKPOINT_INTERVAL_TOKENS} \
      --eval-interval-tokens ${EVAL_INTERVAL_TOKENS} \
      ${CKPT_S3_ARG} \
      ${VERSION_ARG} \
      ${WANDB_ENTITY_ARG} \
      ${WANDB_EXISTING_ARG} \
      ${RESUME_ARGS}
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
#SBATCH --error=${RUN_DIR}/logs/train-%j.err

export RUN_DIR=${RUN_DIR}
bash "${RUN_DIR}/launch_train.sh"
EOF

JOB_ID=$(sbatch --parsable "${RUN_DIR}/train.sbatch")

echo "job_id=${JOB_ID}"
echo "${JOB_ID}" > "${RUN_DIR}/job_id.txt"
echo "RUN_DIR=${RUN_DIR}"
echo "DATASET_ID=${DATASET_ID}"
echo "DATASET_VERSION=${DATASET_VERSION:-latest}"
echo "STAGE_DIR=${STAGE_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "CHECKPOINT_S3_URI=${CHECKPOINT_S3_URI}"
echo "RESUME_FROM_S3=${RESUME_FROM_S3}"
echo "NUM_NODES=${NUM_NODES}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "SLURM_MEM=${SLURM_MEM}"
echo "RESUME_FROM=${RESUME_FROM}"
echo "NODELIST=${NODELIST}"
echo "NUM_EPOCHS=${NUM_EPOCHS}"
echo "MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "CHECKPOINT_INTERVAL_TOKENS=${CHECKPOINT_INTERVAL_TOKENS}"
echo "EVAL_INTERVAL_TOKENS=${EVAL_INTERVAL_TOKENS}"
echo "submitted smollm2-40ep=${JOB_ID}"
