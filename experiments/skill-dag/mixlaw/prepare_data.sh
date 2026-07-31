#!/usr/bin/env bash
# Recipe sidecars for the 24 mixing-law probes (DataDecide-60M).
#
# Requires a working pool already staged from published edullm-data via
#   stage_working_pool_from_edullm_data.py / submit_mixlaw_pilot_pool.sh
# Sole supported path: DomainMixtureStream at mixtures.json weights
# (no per-mix materialized slices; do not use build_mixture_data.py).
#
# Required:
#   WORK   ephemeral job root (contains pool/; recipes written under recipe/)
#
# Optional:
#   POOL_DIR   default $WORK/pool (must carry edullm_data_source.json)
#   VENV       python env with mixlaw deps (default: active python / $WORK/venv)
#   DATASET_ID default pretrain/olmo-127b
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-${SCRIPT_DIR}}"

: "${WORK:?Set WORK to an ephemeral job root (pool/ + recipe/)}"

POOL_DIR="${POOL_DIR:-$WORK/pool}"
RECIPE_WORK="${RECIPE_WORK:-$WORK/recipe}"
TOKENS_PER_PARAM="${TOKENS_PER_PARAM:-5}"
DATASET_ID="${DATASET_ID:-pretrain/olmo-127b}"

log() { echo "[$(date -Is)] $*"; }

if [[ -n "${VENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
elif [[ -x "$WORK/venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$WORK/venv/bin/activate"
fi
# shellcheck disable=SC1091
[[ -f "$WORK/env.sh" ]] && source "$WORK/env.sh"

if [[ ! -f "$POOL_DIR/edullm_data_source.json" ]]; then
  echo "missing $POOL_DIR/edullm_data_source.json — refuse orphan pool; stage from edullm-data first" >&2
  echo "  (submit_mixlaw_pilot_pool.sh or stage_working_pool_from_edullm_data.py)" >&2
  exit 2
fi
if [[ ! -d "$POOL_DIR/tokenized/dclm" && ! -f "$POOL_DIR/dclm/dclm.npy" ]]; then
  echo "missing domain memmaps under $POOL_DIR" >&2
  exit 2
fi

mkdir -p "$RECIPE_WORK"
log "writing recipe sidecars (tokens/param=$TOKENS_PER_PARAM dataset_id=$DATASET_ID)"
python "$CODE_DIR/prepare_mixlaw_pilot_data.py" \
  --work "$RECIPE_WORK" \
  --tokens-per-param "$TOKENS_PER_PARAM" \
  --dataset-id "$DATASET_ID"

log "recipe sidecars ready under $RECIPE_WORK"
log "train with POOL_DIR=$POOL_DIR MIX_WEIGHTS_JSON=$RECIPE_WORK/mix01/mix_weights.json"
