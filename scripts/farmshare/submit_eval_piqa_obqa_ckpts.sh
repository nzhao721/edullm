#!/usr/bin/env bash
# Submit one single-GPU job per (checkpoint × task) for PIQA + OpenBookQA
# on smollm2-135m-750m-27ep-fresh, logging into the original W&B run.
#
# FarmShare gpu QoS MaxSubmitPU=32, so we submit one array per task (18 each)
# instead of a single 36-task array.
set -Eeuo pipefail

RUN_DIR="${RUN_DIR:-/scratch/users/nzhao2/agent-runs/smollm2-135m-750m-27ep-fresh}"
CKPT_DIR="${CKPT_DIR:-${RUN_DIR}/output/checkpoints}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/output/evals_piqa_obqa}"
STAGING="${STAGING:-/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging}"
VENV="${VENV:-/scratch/users/nzhao2/agent-runs/venvs/smollm2-train}"
EVAL_PY="${EVAL_PY:-${STAGING}/scripts/farmshare/eval_arc_task_loss_smollm.py}"
TASKS_CSV="${TASKS_CSV:-PIQA,OpenBookQA}"
N_SHOT="${N_SHOT:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-edullm-smollm2}"
WANDB_ENTITY="${WANDB_ENTITY:-eduLLM}"
WANDB_RUN_ID="${WANDB_RUN_ID:-$(tr -d ' \t\r\n' < "${RUN_DIR}/output/wandb_run_id.txt")}"
# GPU evals write JSON only by default; use log_offline_evals_to_wandb.py on CPU.
LOG_WANDB="${LOG_WANDB:-0}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
SLURM_MEM="${SLURM_MEM:-24G}"
TIME_LIMIT="${TIME_LIMIT:-0:20:00}"

mkdir -p "${RUN_DIR}/logs" "${OUT_DIR}" "${RUN_DIR}/hf-datasets"

mapfile -t CKPTS < <(ls -1d "${CKPT_DIR}"/step* | sort)
IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
N_CKPT=${#CKPTS[@]}
if (( N_CKPT < 1 )); then
  echo "no checkpoints found under ${CKPT_DIR}" >&2
  exit 2
fi

if [[ ! -f "${RUN_DIR}/wandb-session.env" ]]; then
  echo "missing ${RUN_DIR}/wandb-session.env" >&2
  exit 2
fi
if [[ ! -f "${EVAL_PY}" ]]; then
  echo "missing ${EVAL_PY}" >&2
  exit 2
fi
if [[ -z "${WANDB_RUN_ID}" ]]; then
  echo "WANDB_RUN_ID empty" >&2
  exit 2
fi

submit_task_array() {
  local TASK="$1"
  local TABLE="${RUN_DIR}/logs/eval_${TASK,,}_job_table.tsv"
  local SBATCH="${RUN_DIR}/eval_${TASK,,}.sbatch"
  : > "${TABLE}"
  local ckpt
  for ckpt in "${CKPTS[@]}"; do
    printf '%s\n' "${ckpt}" >> "${TABLE}"
  done

  cat > "${SBATCH}" <<EOF
#!/bin/bash
#SBATCH --job-name=sm2-${TASK,,}
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --exclude=wheat-01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --array=0-$((N_CKPT - 1))%4
#SBATCH --chdir=${RUN_DIR}
#SBATCH --output=${RUN_DIR}/logs/eval-${TASK,,}-%A_%a.out
#SBATCH --error=${RUN_DIR}/logs/eval-${TASK,,}-%A_%a.err

set -Eeuo pipefail
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load python/3.12.3 2>/dev/null || true
module load cuda/12.9.0 2>/dev/null || module load cuda/12.4.0 2>/dev/null || true
source "${VENV}/bin/activate"
# shellcheck disable=SC1091
source "${RUN_DIR}/wandb-session.env"

export HF_HOME="${HF_HOME:-/scratch/users/nzhao2/.cache/huggingface}"
export TRANSFORMERS_CACHE="\${HF_HOME}/hub"
export HF_DATASETS_CACHE="${RUN_DIR}/hf-datasets"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_START_METHOD=thread
mkdir -p "\${HF_DATASETS_CACHE}" "${OUT_DIR}"

IDX="\${SLURM_ARRAY_TASK_ID}"
CKPT="\$(sed -n "\$((IDX + 1))p" "${TABLE}")"
STEP_NAME="\$(basename "\${CKPT}")"
STEP_NUM="\${STEP_NAME#step}"
OUT="${OUT_DIR}/\${STEP_NAME}_${TASK,,}.json"

echo "=== \$(date -Is) array=\${IDX} ckpt=\${STEP_NAME} task=${TASK} ==="
nvidia-smi -L || true

if [[ -f "\${OUT}" && "\${FORCE_REEVAL:-0}" != "1" ]]; then
  echo "skip existing \${OUT}"
  exit 0
fi

WANDB_ARGS=()
if [[ "${LOG_WANDB}" == "1" ]]; then
  WANDB_ARGS=(
    --wandb-project "${WANDB_PROJECT}"
    --wandb-entity "${WANDB_ENTITY}"
    --wandb-run-id "${WANDB_RUN_ID}"
    --wandb-step "\${STEP_NUM}"
    --wandb-lock "${RUN_DIR}/logs/wandb_eval.lock"
  )
fi

python "${EVAL_PY}" \\
  --checkpoint "\${CKPT}" \\
  --tokenizer-id HuggingFaceTB/SmolLM2-135M \\
  --out "\${OUT}" \\
  --run-name "smollm2-135m-750m-27ep-fresh" \\
  --n-shot ${N_SHOT} \\
  --tasks "${TASK}" \\
  "\${WANDB_ARGS[@]}"

echo "=== \$(date -Is) done \${STEP_NAME} ${TASK} ==="
EOF

  local JOB_ID
  JOB_ID=$(sbatch --parsable "${SBATCH}")
  echo "task=${TASK} job_id=${JOB_ID} n=${N_CKPT}"
  echo "${JOB_ID}" > "${RUN_DIR}/logs/eval_${TASK,,}_array_job_id.txt"
}

echo "WANDB_RUN_ID=${WANDB_RUN_ID}"
echo "OUT_DIR=${OUT_DIR}"
echo "N_CKPT=${N_CKPT}"
for task in "${TASKS[@]}"; do
  submit_task_array "${task}"
done
