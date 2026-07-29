#!/usr/bin/env bash
# Submit task-loss jobs that were skipped due to QOS submit limits.
set -Eeuo pipefail

RUN_DIR="${RUN_DIR:-/scratch/users/nzhao2/agent-runs/task-loss-per-ckpt-20260728T003628}"
SUNET="${SUNET:-nzhao2}"
SCRATCH="/scratch/users/${SUNET}"

VENV="${VENV:-${SCRATCH}/agent-runs/olmo-ladder-370m-20260722-185217/venv}"
BASE_CONFIG="${BASE_CONFIG:-${SCRATCH}/agent-runs/olmo-ladder-370m-20260722-185217/checkpoints/edullm-370M-30B/step5000-unsharded/config.yaml}"
CE_ROOT="${CE_ROOT:-${SCRATCH}/checkpoints/token-selection-370m/ce-regmix/checkpoints}"
REFHQ_ROOT="${REFHQ_ROOT:-${SCRATCH}/agent-runs/refhq-models-all-20260727T220851Z/unsharded}"
RELEMA_ROOT="${RELEMA_ROOT:-${SCRATCH}/agent-runs/relema-models-all-20260727T220851Z/unsharded}"

EVAL_TIME="${EVAL_TIME:-00:45:00}"
GATHER_TIME="${GATHER_TIME:-00:30:00}"

job_queued() {
  local name="$1"
  squeue -u "${SUNET}" -h -o "%j" 2>/dev/null | grep -qx "${name}"
}

submit_gather() {
  local model="$1" step="$2" ckpt="$3"
  local out="${ckpt}/model_eval.pt"
  if [[ -f "${out}" ]]; then
    if "${VENV}/bin/python" - <<PY
import sys, torch
from pathlib import Path
p = Path("${out}")
obj = torch.load(p, map_location="cpu", weights_only=False)
sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
if tuple(sd["embeddings.weight"].shape) != (100352, 1024):
    raise SystemExit(1)
print(f"valid {p}", file=sys.stderr)
PY
    then
      return 0
    fi
    rm -f "${out}"
  fi
  if job_queued "g-${model}-s${step}"; then
    echo "SKIP gather queued g-${model}-s${step}" >&2
    return 0
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
  local jname="tl-${model}-s${step}"
  local out="${RUN_DIR}/eval/${model}/step${step}_task_loss.json"
  if [[ -f "${out}" ]]; then
    echo "SKIP existing ${out}" >&2
    return 0
  fi
  if job_queued "${jname}"; then
    echo "SKIP queued ${jname}" >&2
    return 0
  fi
  local dep_flag=()
  if [[ -n "${dep}" ]]; then
    dep_flag=(--dependency=afterok:"${dep}")
  fi
  local job
  if ! job=$(sbatch --exclude=wheat-01 "${dep_flag[@]}" \
    --job-name="${jname}" \
    --time="${EVAL_TIME}" \
    --chdir="${RUN_DIR}" \
    --export=ALL,RUN_DIR="${RUN_DIR}",MODEL_NAME="${model}",STEP="${step}",CHECKPOINT_DIR="${ckpt}",CKPT_FORMAT="${fmt}",VENV="${VENV}",BASE_CONFIG="${BASE_CONFIG}",DEVICE_EVAL_BATCH_SIZE=4 \
    "${RUN_DIR}/scripts/task_loss_eval_step.sbatch" 2>&1 | tee /dev/stderr | awk '/Submitted batch job/{print $NF}'); then
    echo "FAIL ${jname}" >&2
    return 1
  fi
  if [[ -z "${job}" ]]; then
    echo "FAIL ${jname} (no job id)" >&2
    return 1
  fi
  echo "${job}" >> "${RUN_DIR}/eval_jobs.txt"
  echo "SUBMITTED ${jname} job=${job}"
}

n_ok=0
n_fail=0

# CE step2384 gather + eval
ckpt="${CE_ROOT}/step2384"
gather_job=""
if gather_job=$(submit_gather ce 2384 "${ckpt}"); then
  [[ -n "${gather_job}" ]] && echo "${gather_job}" >> "${RUN_DIR}/gather_jobs.txt"
else
  gather_job=""
fi
if submit_eval ce-regmix 2384 "${ckpt}" state_pt "${gather_job}"; then n_ok=$((n_ok + 1)); else n_fail=$((n_fail + 1)); fi

for step in $(find "${REFHQ_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step*' | sed 's#.*/step##' | sort -n); do
  if submit_eval refhq "${step}" "${REFHQ_ROOT}/step${step}" model_pt ""; then
    n_ok=$((n_ok + 1))
  else
    n_fail=$((n_fail + 1))
  fi
done

for step in $(find "${RELEMA_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step*' | sed 's#.*/step##' | sort -n); do
  if submit_eval rel-ema "${step}" "${RELEMA_ROOT}/step${step}" model_pt ""; then
    n_ok=$((n_ok + 1))
  else
    n_fail=$((n_fail + 1))
  fi
done

echo "submitted_ok=${n_ok} failed=${n_fail}"
squeue -u "${SUNET}" -o '%.18i %.20j %.8T %R' | head -50
