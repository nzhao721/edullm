#!/usr/bin/env bash
# Print a plan JSON (no --launch) for the RefHQ-seeded REL arm.
# Requires REF_PT (or a non-null reference.load_path in the YAML) so
# validate_scratch_config can pass for ema.seed_mode=refhq.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TS_ROOT="$(cd "${ARM_DIR}/.." && pwd)"

export PYTHONPATH="${TS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CFG="${CFG:-${ARM_DIR}/configs/run_rel_ema_refhq_10b.yaml}"
OLMO_ROOT="${OLMO_ROOT:-}"
REF_PT="${REF_PT:-}"
EXTRA_ARGS=()
if [[ -n "${OLMO_ROOT}" ]]; then
  EXTRA_ARGS+=(--olmo-root "${OLMO_ROOT}")
fi

TMP_CFG=""
cleanup() {
  if [[ -n "${TMP_CFG}" && -f "${TMP_CFG}" ]]; then
    rm -f "${TMP_CFG}"
  fi
}
trap cleanup EXIT

if [[ -n "${REF_PT}" ]]; then
  TMP_CFG="$(mktemp "${TMPDIR:-/tmp}/rel_ema_refhq_plan_XXXXXX.yaml")"
  REF_PT="${REF_PT}" CFG="${CFG}" TMP_CFG="${TMP_CFG}" python - <<'PY'
import os
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(os.environ["CFG"]).read_text(encoding="utf-8"))
cfg.setdefault("reference", {})["load_path"] = os.environ["REF_PT"]
Path(os.environ["TMP_CFG"]).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
  CFG="${TMP_CFG}"
fi

exec python -m token_selection.scripts.train_olmo_template \
  --config "${CFG}" \
  --method rel_ema \
  "${EXTRA_ARGS[@]}"
