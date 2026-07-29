#!/usr/bin/env bash
# Hardware-agnostic launch for learnability-token (1..N GPUs via torchrun).
#
# Required env:
#   OLMO_ROOT   — pinned edu-llm/OLMo-core checkout (revision in the YAML)
#
# Optional env:
#   NPROC / CUDA_VISIBLE_DEVICES — world size discovery (same as attention/)
#   CONFIG or CFG                — override YAML path
#   RESUME=1                     — resume from latest matching fingerprint
#
# Prerequisites: export early/late .pt (export_learnability_refs.py) and set
# reference.early/late.load_path in the YAML (null until then; fail-closed).
#
# Examples (from experiments/token-selection/):
#   CUDA_VISIBLE_DEVICES=0 NPROC=1 bash learnability-token/launch.sh
#   CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash learnability-token/launch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM_ROOT="${SCRIPT_DIR}"
TS_ROOT="$(cd "${ARM_ROOT}/.." && pwd)"
export PYTHONPATH="${TS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-${CFG:-${ARM_ROOT}/configs/run_learnability_10b.yaml}}"
METHOD="${METHOD:-learnability}"

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

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export TOKEN_SELECTION_SKIP_IDLE_CHECK="${TOKEN_SELECTION_SKIP_IDLE_CHECK:-1}"
fi

LAUNCH_ARGS=(--config "${CONFIG}" --method "${METHOD}" --olmo-root "${OLMO_ROOT}" --launch)
if [[ "${RESUME:-0}" == "1" ]]; then
  LAUNCH_ARGS+=(--resume)
fi

echo "[learnability-token] nproc=${NPROC} config=${CONFIG} method=${METHOD}"
cd "${TS_ROOT}"
exec python -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
  -m token_selection.scripts.train_olmo_template \
  "${LAUNCH_ARGS[@]}"
