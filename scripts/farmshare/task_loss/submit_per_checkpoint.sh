#!/usr/bin/env bash
# Submit one single-GPU task-loss job per checkpoint (BLADE, CE-RegMix, RefHQ, REL-EMA).
# CE/BLADE legacy state.pt checkpoints get a short 2-GPU gather job first.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
SCRATCH="/scratch/users/${SUNET}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%S)}"
RUN_DIR="${RUN_DIR:-${SCRATCH}/agent-runs/task-loss-per-ckpt-${STAMP}}"

VENV="${VENV:-${SCRATCH}/agent-runs/olmo-ladder-370m-20260722-185217/venv}"
BASE_CONFIG="${BASE_CONFIG:-${SCRATCH}/agent-runs/olmo-ladder-370m-20260722-185217/checkpoints/edullm-370M-30B/step5000-unsharded/config.yaml}"

# Local checkpoint roots (no AWS at eval time).
BLADE_ROOT="${BLADE_ROOT:-${SCRATCH}/checkpoints/token-selection-370m/blade/checkpoints}"
CE_ROOT="${CE_ROOT:-${SCRATCH}/checkpoints/token-selection-370m/ce-regmix/checkpoints}"
REFHQ_ROOT="${REFHQ_ROOT:-${SCRATCH}/agent-runs/refhq-models-all-20260727T220851Z/unsharded}"
RELEMA_ROOT="${RELEMA_ROOT:-${SCRATCH}/agent-runs/relema-models-all-20260727T220851Z/unsharded}"

EVAL_TIME="${EVAL_TIME:-00:45:00}"
GATHER_TIME="${GATHER_TIME:-00:30:00}"

SCRIPT_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SCRIPTS="$(cd "${SCRIPT_SRC}/.." && pwd)"

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${RUN_DIR}/eval"
chmod 700 "${RUN_DIR}"

if [[ "${SKIP_SCRIPT_COPY:-0}" != "1" && "${SCRIPT_SRC}" != "${RUN_DIR}/scripts" ]]; then
  cp -f "${SCRIPT_SRC}/eval_task_loss_olmo_core.py" "${RUN_DIR}/scripts/"
  cp -f "${SCRIPT_SRC}/prepare_model_eval_pt.py" "${RUN_DIR}/scripts/"
  cp -f "${SCRIPT_SRC}/task_loss_eval_step.sbatch" "${RUN_DIR}/scripts/"
  cp -f "${SCRIPT_SRC}/gather_hsdp_step.sbatch" "${RUN_DIR}/scripts/"
  cp -f "${REPO_SCRIPTS}/gather_hsdp_state_pt_to_model.py" "${RUN_DIR}/scripts/"
  sed -i 's/\r$//' "${RUN_DIR}/scripts/"*.{py,sbatch,sh} 2>/dev/null || true
  chmod +x "${RUN_DIR}/scripts/"*.sbatch
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "missing venv ${VENV}" >&2
  exit 1
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "missing base config ${BASE_CONFIG}" >&2
  exit 1
fi

if ! "${VENV}/bin/python" -c 'import olmo_core, olmo, datasets' 2>/dev/null; then
  echo "venv missing olmo_core/olmo/datasets: ${VENV}" >&2
  exit 1
fi

list_steps() {
  local root="$1"
  find "${root}" -mindepth 1 -maxdepth 1 -type d -name 'step*' \
    | sed 's#.*/##' \
    | sed 's/^step//' \
    | sort -n
}

submit_gather() {
  local model="$1" step="$2" ckpt="$3"
  local out="${ckpt}/model_eval.pt"
  if [[ -f "${out}" ]]; then
    if "${VENV}/bin/python" - <<PY
import sys
import torch
from pathlib import Path
p = Path("${out}")
obj = torch.load(p, map_location="cpu", weights_only=False)
sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
emb = tuple(sd["embeddings.weight"].shape)
print(f"existing {p} emb {emb}", file=sys.stderr)
if emb != (100352, 1024):
    raise SystemExit(1)
PY
    then
      echo "SKIP gather ${model} step${step} (valid ${out})" >&2
      return 0
    fi
    echo "regather ${model} step${step} (bad ${out})" >&2
    rm -f "${out}"
  fi
  sbatch --exclude=wheat-01 \
    --job-name="g-${model}-s${step}" \
    --time="${GATHER_TIME}" \
    --chdir="${RUN_DIR}" \
    --export=ALL,RUN_DIR="${RUN_DIR}",CHECKPOINT_DIR="${ckpt}",OUT_PT="${out}",VENV="${VENV}" \
    "${RUN_DIR}/scripts/gather_hsdp_step.sbatch" | awk '{print $NF}'
}

submit_eval() {
  local model="$1" step="$2" ckpt="$3" fmt="$4" dep="${5:-}"
  local dep_flag=()
  if [[ -n "${dep}" ]]; then
    dep_flag=(--dependency=afterok:"${dep}")
  fi
  local job
  job=$(sbatch --exclude=wheat-01 "${dep_flag[@]}" \
    --job-name="tl-${model}-s${step}" \
    --time="${EVAL_TIME}" \
    --chdir="${RUN_DIR}" \
    --export=ALL,RUN_DIR="${RUN_DIR}",MODEL_NAME="${model}",STEP="${step}",CHECKPOINT_DIR="${ckpt}",CKPT_FORMAT="${fmt}",VENV="${VENV}",BASE_CONFIG="${BASE_CONFIG}",DEVICE_EVAL_BATCH_SIZE=4 \
    "${RUN_DIR}/scripts/task_loss_eval_step.sbatch" | awk '{print $NF}')
  echo "${job}" >> "${RUN_DIR}/eval_jobs.txt"
  echo "SUBMITTED tl-${model}-s${step} job=${job} dep=${dep:-none} ckpt=${ckpt}"
}

: > "${RUN_DIR}/eval_jobs.txt"
: > "${RUN_DIR}/gather_jobs.txt"

n_gather=0
n_eval=0

for step in $(list_steps "${BLADE_ROOT}"); do
  ckpt="${BLADE_ROOT}/step${step}"
  gather_job=""
  if ! gather_job=$(submit_gather blade "${step}" "${ckpt}"); then
    gather_job=""
  fi
  if [[ -n "${gather_job}" ]]; then
    echo "${gather_job}" >> "${RUN_DIR}/gather_jobs.txt"
    n_gather=$((n_gather + 1))
  fi
  submit_eval blade "${step}" "${ckpt}" state_pt "${gather_job}"
  n_eval=$((n_eval + 1))
done

for step in $(list_steps "${CE_ROOT}"); do
  ckpt="${CE_ROOT}/step${step}"
  gather_job=""
  if ! gather_job=$(submit_gather ce "${step}" "${ckpt}"); then
    gather_job=""
  fi
  if [[ -n "${gather_job}" ]]; then
    echo "${gather_job}" >> "${RUN_DIR}/gather_jobs.txt"
    n_gather=$((n_gather + 1))
  fi
  submit_eval ce-regmix "${step}" "${ckpt}" state_pt "${gather_job}"
  n_eval=$((n_eval + 1))
done

for step in $(list_steps "${REFHQ_ROOT}"); do
  ckpt="${REFHQ_ROOT}/step${step}"
  submit_eval refhq "${step}" "${ckpt}" model_pt ""
  n_eval=$((n_eval + 1))
done

for step in $(list_steps "${RELEMA_ROOT}"); do
  ckpt="${RELEMA_ROOT}/step${step}"
  submit_eval rel-ema "${step}" "${ckpt}" model_pt ""
  n_eval=$((n_eval + 1))
done

echo "RUN_DIR=${RUN_DIR}"
echo "gather_jobs=${n_gather} eval_jobs=${n_eval}"
squeue -u "${SUNET}" -o '%.18i %.12P %.20j %.8T %.10M %.6D %R' || true
