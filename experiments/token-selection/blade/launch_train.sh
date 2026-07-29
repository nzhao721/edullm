#!/usr/bin/env bash
# Hardware-agnostic BLADE launch. World size comes from torchrun / env — never hardcode GPUs.
#
# Usage:
#   # Single GPU
#   bash experiments/token-selection/blade/launch_train.sh \
#     --name blade-regmix10b-v2 \
#     --train-paths-file /data/blade/train_tokenized/paths_train.txt \
#     --ref-paths-file /data/blade/ref_tokenized/paths_refhq.txt \
#     --save-folder /data/ckpts/blade-regmix10b-v2 \
#     --progress-dir /data/runs/blade-regmix10b-v2 \
#     --length-tokens 10000058051 \
#     --fresh
#
#   # Multi-GPU (example: 4 processes)
#   NPROC_PER_NODE=4 bash experiments/token-selection/blade/launch_train.sh ...
#
# Optional env:
#   NPROC_PER_NODE   default: 1 (or count of CUDA_VISIBLE_DEVICES if set)
#   MASTER_ADDR / MASTER_PORT
#   TASK_LOSS_EVAL=0 to disable post-save eval spawn
#   TASK_LOSS_EVAL_SCRIPT / TASK_LOSS_CUDA_VISIBLE_DEVICES
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/experiments/token-selection${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  NPROC="${NPROC_PER_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # Count comma-separated device ids (empty → 0 → fall back to 1).
  IFS=',' read -r -a _devs <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC="${#_devs[@]}"
  [[ "${NPROC}" -ge 1 ]] || NPROC=1
else
  NPROC=1
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

echo "launch_train: nproc_per_node=${NPROC} master=${MASTER_ADDR}:${MASTER_PORT}"
exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/train_blade_olmo_370m.py" \
  "$@"
