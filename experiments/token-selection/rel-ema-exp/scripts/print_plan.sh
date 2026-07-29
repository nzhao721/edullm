#!/usr/bin/env bash
# Print a plan JSON (no --launch) for the REL no-init exp-α arm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TS_ROOT="$(cd "${ARM_DIR}/.." && pwd)"

export PYTHONPATH="${TS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CFG="${CFG:-${ARM_DIR}/configs/run_rel_ema_exp_10b.yaml}"
OLMO_ROOT="${OLMO_ROOT:-}"
EXTRA_ARGS=()
if [[ -n "${OLMO_ROOT}" ]]; then
  EXTRA_ARGS+=(--olmo-root "${OLMO_ROOT}")
fi

exec python -m token_selection.scripts.train_olmo_template \
  --config "${CFG}" \
  --method rel_ema \
  "${EXTRA_ARGS[@]}"
