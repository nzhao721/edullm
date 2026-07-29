#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=tl-eval
#SBATCH --output=logs/task_loss_%x_%j.out
#SBATCH --error=logs/task_loss_%x_%j.err

# One model family per job. Loops every step checkpoint under CKPT_ROOT.
# Required exports: RUN_DIR, MODEL_NAME, CKPT_ROOT, CKPT_FORMAT, VENV
set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${MODEL_NAME:?}"
: "${CKPT_ROOT:?}"
: "${CKPT_FORMAT:?}"   # state_pt | distcp
: "${VENV:?}"

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
DEVICE_BS="${DEVICE_EVAL_BATCH_SIZE:-4}"

mapfile -t STEPS < <(
  find "${CKPT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step*' \
    | sed 's#.*/##' \
    | sed 's/^step//' \
    | sort -n
)

if [[ ${#STEPS[@]} -eq 0 ]]; then
  echo "No step* dirs under ${CKPT_ROOT}" >&2
  exit 1
fi

echo "model=${MODEL_NAME} format=${CKPT_FORMAT} n_steps=${#STEPS[@]} root=${CKPT_ROOT}"
echo "host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"

failed=0
for step in "${STEPS[@]}"; do
  ckpt="${CKPT_ROOT}/step${step}"
  out="${RUN_DIR}/eval/${MODEL_NAME}/step${step}_task_loss.json"
  if [[ -f "${out}" ]]; then
    echo "SKIP step${step} (exists: ${out})"
    continue
  fi
  echo "=== EVAL ${MODEL_NAME} step${step} ==="
  extra=()
  if [[ -n "${BASE_CONFIG}" && -f "${BASE_CONFIG}" ]]; then
    extra+=(--base-config "${BASE_CONFIG}")
  fi
  # Unshard distcp outside torchrun: olmo-core forbids unshard_checkpoint
  # once a distributed / torchrun context is active.
  if [[ "${CKPT_FORMAT}" == "distcp" && ! -f "${ckpt}/model_eval.pt" ]]; then
    echo "Materializing ${ckpt}/model_eval.pt (plain python, no torchrun)"
    if ! python - <<PY
from pathlib import Path
import importlib.util
p = Path("${EVAL_PY}")
spec = importlib.util.spec_from_file_location("eval_task_loss_olmo_core", p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.materialize_distcp_model_eval(Path("${ckpt}"))
print("materialized", Path("${ckpt}") / "model_eval.pt")
PY
    then
      echo "FAIL materialize step${step}" >&2
      failed=$((failed + 1))
      continue
    fi
  fi
  fmt_for_eval="${CKPT_FORMAT}"
  if [[ "${CKPT_FORMAT}" == "distcp" ]]; then
    fmt_for_eval="state_pt"
  fi
  if ! torchrun --standalone --nproc_per_node=1 \
      "${EVAL_PY}" \
      --checkpoint "${ckpt}" \
      --format "${fmt_for_eval}" \
      --out "${out}" \
      --run-name "${MODEL_NAME}-step${step}" \
      --device-eval-batch-size "${DEVICE_BS}" \
      "${extra[@]}"; then
    echo "FAIL step${step}" >&2
    failed=$((failed + 1))
    continue
  fi
  echo "OK step${step} -> ${out}"
done

echo "done model=${MODEL_NAME} failures=${failed}"
if [[ "${failed}" -eq "${#STEPS[@]}" ]]; then
  exit 1
fi
exit 0
