#!/usr/bin/env bash
# Write Skill-It probe recipe sidecars (no slice materialization).
#
# Required:
#   POOL_DIR   edullm-data working pool (training uses same pool; marker optional here)
#
# Optional:
#   SKILLIT_PROBE_WORK  recipe sidecar root (default: $WORK/skillit-probes)
#   OUT_DIR             alias for recipe work root (default: $SKILLIT_PROBE_WORK/recipe)
#   PROBES_JSON         default: sibling probes.json
#   TOKENS_PER_PARAM    default 5
#
# Example:
#   POOL_DIR=$WORK/pool OUT_DIR=$WORK/skillit-probes/recipe \
#     bash prepare_probes.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBES_JSON="${PROBES_JSON:-${SCRIPT_DIR}/probes.json}"
TOKENS_PER_PARAM="${TOKENS_PER_PARAM:-5}"
PYTHON="${PYTHON:-python}"
if [[ -z "${SKILLIT_PROBE_WORK:-}" ]]; then
  : "${WORK:?set SKILLIT_PROBE_WORK or WORK}"
  SKILLIT_PROBE_WORK="${WORK%/}/skillit-probes"
fi
OUT_DIR="${OUT_DIR:-${SKILLIT_PROBE_WORK}/recipe}"

: "${POOL_DIR:?POOL_DIR is required (olmohq working pool)}"

echo "[prepare_probes] recipe sidecars from ${PROBES_JSON} tpp=${TOKENS_PER_PARAM}"
"${PYTHON}" "${SCRIPT_DIR}/prepare_skillit_probe_data.py" \
  --recipe "${PROBES_JSON}" \
  --work "${OUT_DIR}" \
  --tokens-per-param "${TOKENS_PER_PARAM}"

echo "[prepare_probes] done → ${OUT_DIR}"
