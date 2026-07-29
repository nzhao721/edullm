#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=wheat-01
#SBATCH --job-name=tl-step
#SBATCH --output=logs/tl_%x_%j.out
#SBATCH --error=logs/tl_%x_%j.err

# Single-checkpoint RC 5-shot task-loss eval (one GPU).
# Required exports: RUN_DIR, MODEL_NAME, STEP, CHECKPOINT_DIR, CKPT_FORMAT, VENV
set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${MODEL_NAME:?}"
: "${STEP:?}"
: "${CHECKPOINT_DIR:?}"
: "${CKPT_FORMAT:?}"   # state_pt | distcp | model_pt
: "${VENV:?}"

TIME_LIMIT="${SLURM_TIME_LIMIT:-}"
if [[ -z "${TIME_LIMIT}" ]]; then
  echo "WARN: set --time on sbatch (e.g. 00:45:00)" >&2
fi

cd "${RUN_DIR}"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/eval/${MODEL_NAME}"

# shellcheck disable=SC1090
source "${VENV}/bin/activate"

export PYTHONUNBUFFERED=1
export WANDB_DISABLED=1
export WANDB_MODE=disabled
export HF_HOME="${HF_HOME:-${RUN_DIR}/hf_cache}"
export OLMES_DATA_DIR="${OLMES_DATA_DIR:-${RUN_DIR}/olmes_cache}"
mkdir -p "${HF_HOME}" "${OLMES_DATA_DIR}"

BASE_CONFIG="${BASE_CONFIG:-}"
EVAL_PY="${RUN_DIR}/scripts/eval_task_loss_olmo_core.py"
PREP_PY="${RUN_DIR}/scripts/prepare_model_eval_pt.py"
DEVICE_BS="${DEVICE_EVAL_BATCH_SIZE:-4}"

ckpt="${CHECKPOINT_DIR}"
out="${RUN_DIR}/eval/${MODEL_NAME}/step${STEP}_task_loss.json"

if [[ -f "${out}" ]]; then
  echo "SKIP ${MODEL_NAME} step${STEP} (exists ${out})"
  exit 0
fi

echo "model=${MODEL_NAME} step=${STEP} format=${CKPT_FORMAT} ckpt=${ckpt}"
echo "host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ "${CKPT_FORMAT}" == "model_pt" ]]; then
  if [[ ! -f "${ckpt}/model_eval.pt" ]]; then
    python "${PREP_PY}" --checkpoint-dir "${ckpt}"
  fi
  fmt_for_eval="state_pt"
elif [[ "${CKPT_FORMAT}" == "distcp" && ! -f "${ckpt}/model_eval.pt" ]]; then
  echo "Materializing ${ckpt}/model_eval.pt (plain python, no torchrun)"
  python - <<PY
from pathlib import Path
import importlib.util
p = Path("${EVAL_PY}")
spec = importlib.util.spec_from_file_location("eval_task_loss_olmo_core", p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.materialize_distcp_model_eval(Path("${ckpt}"))
print("materialized", Path("${ckpt}") / "model_eval.pt")
PY
  fmt_for_eval="state_pt"
elif [[ "${CKPT_FORMAT}" == "distcp" ]]; then
  fmt_for_eval="state_pt"
else
  fmt_for_eval="${CKPT_FORMAT}"
fi

extra=()
if [[ -n "${BASE_CONFIG}" && -f "${BASE_CONFIG}" ]]; then
  extra+=(--base-config "${BASE_CONFIG}")
fi

torchrun --standalone --nproc_per_node=1 \
  "${EVAL_PY}" \
  --checkpoint "${ckpt}" \
  --format "${fmt_for_eval}" \
  --out "${out}" \
  --run-name "${MODEL_NAME}-step${STEP}" \
  --device-eval-batch-size "${DEVICE_BS}" \
  "${extra[@]}"

echo "OK ${MODEL_NAME} step${STEP} -> ${out}"
