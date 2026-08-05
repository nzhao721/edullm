#!/usr/bin/env bash
# Poll live MixLaw training progress on a RunPod host.
# Usage: mixlaw_poll_run_status.sh <arm_name>
# Example: mixlaw_poll_run_status.sh mix01
set -euo pipefail

ARM_NAME="${1:?arm name required}"
RUN_ROOT="${RUN_ROOT:-/workspace/edullm-runs/mixlaw}"
ARM_ROOT="${RUN_ROOT}/${ARM_NAME}"
LOG=""

if [[ -f "${ARM_ROOT}/launch.out" ]]; then
  LOG="${ARM_ROOT}/launch.out"
elif [[ -f /workspace/mixlaw-train.log ]]; then
  LOG="/workspace/mixlaw-train.log"
fi

CKPT_STEP=""
if [[ -d "${ARM_ROOT}/checkpoints" ]]; then
  CKPT_STEP="$(ls "${ARM_ROOT}/checkpoints" 2>/dev/null | grep '^step' | sort -V | tail -1 | sed 's/^step//')"
fi

PHASE="stopped"
if pgrep -f "entrypoint.py" >/dev/null 2>&1; then
  PHASE="train"
fi

EVAL_DONE="0"
if [[ -f "${ARM_ROOT}/eval-work/task-loss/step2384_task_loss.json" ]]; then
  EVAL_DONE="1"
  if [[ "${PHASE}" == "stopped" ]]; then
    PHASE="done"
  fi
fi

echo "arm=${ARM_NAME}"
echo "phase=${PHASE}"
echo "ckpt_step=${CKPT_STEP:-none}"
echo "eval2384=${EVAL_DONE}"
echo "log=${LOG:-none}"
if [ -n "${LOG}" ]; then
  grep -E '\[step=|throughput/device/TPS' "${LOG}" | tail -n 120
fi
