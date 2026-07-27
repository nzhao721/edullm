#!/usr/bin/env bash
# One-time data preparation for the 24 mixing-law probes.
#
# Source: s3://edullm-dataset-olmohq/olmo-mix-1124-30b (raw json.gz shards).
# Does NOT sync the full ~130 GiB tree. Instead:
#   1. pulls the tiny plan/manifest.jsonl inventory
#   2. randomly selects only enough shards to cover peak demand at TOKENS_PER_PARAM
#   3. downloads those shards (~a few GiB at the default budget)
#   4. tokenizes a working pool from them
#   5. plans and materializes the 24 per-mixture memmap slices
#
# Default TOKENS_PER_PARAM=5 is sized for ≈12 B200 GPU-hours across all 24
# mixtures (see budget_calculator.py).
#
# Run once on shared storage every compute node can read. No GPUs required.
set -euo pipefail

WORK="${WORK:-/opt/edullm/mixlaw}"
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV="${VENV:-$WORK/venv}"
OLMOHQ_S3="${OLMOHQ_S3:-s3://edullm-dataset-olmohq/olmo-mix-1124-30b}"
RAW_DIR="${RAW_DIR:-$WORK/olmohq/data}"
TOKENIZED_DIR="${TOKENIZED_DIR:-$WORK/tokenized}"
SLICE_DIR="${SLICE_DIR:-$WORK/slices}"
PLAN_DIR="${PLAN_DIR:-$WORK/olmohq/plan}"
TOKENS_PER_PARAM="${TOKENS_PER_PARAM:-5}"
BLOCK_SEQS="${BLOCK_SEQS:-256}"
BUILD_WORKERS="${BUILD_WORKERS:-8}"
TOKENIZE_WORKERS="${TOKENIZE_WORKERS:-16}"
FETCH_WORKERS="${FETCH_WORKERS:-8}"
SHARD_OVERSHOOT="${SHARD_OVERSHOOT:-1.5}"
SKIP_FETCH="${SKIP_FETCH:-0}"
SKIP_TOKENIZE="${SKIP_TOKENIZE:-0}"

log() { echo "[$(date -Is)] $*"; }

# shellcheck disable=SC1091
source "$VENV/bin/activate"
# shellcheck disable=SC1091
[[ -f "$WORK/env.sh" ]] && source "$WORK/env.sh"

mkdir -p "$RAW_DIR" "$TOKENIZED_DIR" "$SLICE_DIR" "$PLAN_DIR"

if [[ "$SKIP_FETCH" != "1" ]]; then
  log "fetching shard inventory (manifest.jsonl)"
  aws s3 cp "$OLMOHQ_S3/plan/manifest.jsonl" "$PLAN_DIR/manifest.jsonl" --only-show-errors

  log "selecting + downloading only the shards needed for tokens/param=$TOKENS_PER_PARAM"
  python "$CODE_DIR/select_and_fetch_shards.py" \
    --manifest "$PLAN_DIR/manifest.jsonl" \
    --raw-dir "$RAW_DIR" \
    --out-plan "$PLAN_DIR/shard_selection.json" \
    --tokens-per-param "$TOKENS_PER_PARAM" \
    --overshoot "$SHARD_OVERSHOOT" \
    --fetch-workers "$FETCH_WORKERS"
fi

log "raw domain footprints (selected shards only):"
du -h --max-depth=1 "$RAW_DIR" 2>/dev/null | sort -h || true

if [[ "$SKIP_TOKENIZE" != "1" ]]; then
  log "tokenizing working pool at tokens/param=$TOKENS_PER_PARAM"
  python "$CODE_DIR/tokenize_working_pool.py" \
    --data-dir "$RAW_DIR" \
    --out-dir "$TOKENIZED_DIR" \
    --tokens-per-param "$TOKENS_PER_PARAM" \
    --workers "$TOKENIZE_WORKERS"
fi

log "planning random per-mixture subsamples"
python "$CODE_DIR/build_mixture_data.py" plan \
  --tokenized-dir "$TOKENIZED_DIR" \
  --out-dir "$SLICE_DIR" \
  --tokens-per-param "$TOKENS_PER_PARAM" \
  --block-seqs "$BLOCK_SEQS"

log "materializing slices with $BUILD_WORKERS parallel workers"
python "$CODE_DIR/build_mixture_data.py" build \
  --plan-dir "$SLICE_DIR" \
  --out-dir "$SLICE_DIR" \
  --tokenized-dir "$TOKENIZED_DIR" \
  --workers "$BUILD_WORKERS"

log "verifying every mixture"
python - "$SLICE_DIR" <<'PY'
import json
import sys
from pathlib import Path

slice_dir = Path(sys.argv[1])
plan = json.loads((slice_dir / "slice_plan.json").read_text())
worst = 0.0
for mix in plan["mixtures"]:
    meta_path = slice_dir / mix["run_name"] / "mix_meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text())
    on_disk = sum(Path(p).stat().st_size // 4 for p in meta["paths"])
    if on_disk != meta["tokens"]:
        raise SystemExit(f"{mix['run_name']}: {on_disk} tokens on disk != {meta['tokens']} planned")
    err = max(
        abs(meta["realized_weights"][d] - meta["target_weights"][d]) for d in meta["target_weights"]
    )
    worst = max(worst, err)
print(f"all {len(plan['mixtures'])} mixtures verified; worst weight error {worst:.2e}")
print(f"tokens per mixture: {plan['total_tokens_per_mix']:,} ({plan['total_steps_per_mix']:,} steps)")
PY

log "data ready in $SLICE_DIR"
