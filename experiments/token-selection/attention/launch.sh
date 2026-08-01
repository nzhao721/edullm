#!/usr/bin/env bash
# Launch attention_topk on 1..N GPUs (hardware-agnostic).
#
# Required env:
#   OLMO_ROOT   — pinned edu-llm/OLMo-core checkout (revision in the YAML)
#
# Optional env:
#   NPROC                 — torchrun nproc_per_node (default: # of visible GPUs, else 1)
#   CUDA_VISIBLE_DEVICES  — physical GPU pin (required outside Slurm unless YAML sets it)
#   CONFIG                — override YAML path
#   RESUME=1              — resume from local scratch or WANDB_RESUME_ARTIFACT
#   TOKEN_SELECTION_SKIP_IDLE_CHECK=1 — skip idle GPU probe (Slurm sets this path too)
#
# W&B project token-selection is the artifact and metrics store. Push
# wandb-session.env via scripts/farmshare/push_wandb_session_to_farmshare.sh
# "$RUN_DIR" (or set WANDB_SESSION_ENV). Local smoke: WANDB_MODE=disabled.
#
# Examples (from experiments/token-selection/):
#   # 1 GPU
#   CUDA_VISIBLE_DEVICES=0 NPROC=1 bash attention/launch.sh
#   # 4 GPUs
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 bash attention/launch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM_ROOT="${SCRIPT_DIR}"
TS_ROOT="$(cd "${ARM_ROOT}/.." && pwd)"
export PYTHONPATH="${TS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-${ARM_ROOT}/configs/run_attention_10b.yaml}"
METHOD="${METHOD:-attention_topk}"

if [[ -z "${OLMO_ROOT:-}" ]]; then
  echo "OLMO_ROOT must point at the pinned OLMo-core checkout" >&2
  exit 1
fi

if [[ -z "${NPROC:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _gpus <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC="${#_gpus[@]}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    NPROC="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    NPROC="${NPROC:-1}"
  else
    NPROC=1
  fi
fi
if [[ "${NPROC}" -lt 1 ]]; then
  echo "NPROC must be >= 1 (got ${NPROC})" >&2
  exit 1
fi
export TASK_LOSS_STRICT=1
export TASK_LOSS_NPROC="${TASK_LOSS_NPROC:-${NPROC}}"

LAUNCH_ARGS=(--config "${CONFIG}" --method "${METHOD}" --olmo-root "${OLMO_ROOT}" --launch)
if [[ "${RESUME:-0}" == "1" ]]; then
  LAUNCH_ARGS+=(--resume)
fi
if [[ -n "${WANDB_RESUME_ARTIFACT:-}" ]]; then
  LAUNCH_ARGS+=(--wandb-resume-artifact "${WANDB_RESUME_ARTIFACT}")
fi

echo "attention launch: NPROC=${NPROC} CONFIG=${CONFIG} METHOD=${METHOD}"
_ATTENTION_RUN_NAME="${RUN_NAME:-attention-topk-10b-scratch-v1}"
# shellcheck disable=SC1091
source "${TS_ROOT}/token_selection/scripts/wandb_env.sh" "attention" "${_ATTENTION_RUN_NAME}"
exec python -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
  -m token_selection.scripts.train_olmo_template \
  "${LAUNCH_ARGS[@]}"
