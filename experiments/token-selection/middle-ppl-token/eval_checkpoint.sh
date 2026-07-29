#!/usr/bin/env bash
# Run full 20-label task_loss_bpb on one Middle-PPL token checkpoint.
#
# Usage:
#   bash eval_checkpoint.sh /path/to/.../checkpoints/middle_ppl/step125
#   STEP=125 CKPT_ROOT=/path/to/checkpoints/middle_ppl bash eval_checkpoint.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EDULLM_ROOT="${EDULLM_ROOT:-$(cd "$TS_ROOT/../.." && pwd)}"
EVAL_PY="${TASK_LOSS_EVAL_SCRIPT:-$EDULLM_ROOT/scripts/farmshare/task_loss/eval_task_loss_olmo_core.py}"
RUN_NAME="${RUN_NAME:-middle-ppl-token-10b-v1}"
OUT_DIR="${OUT_DIR:-$TS_ROOT/task_loss_results/middle-ppl-token}"
BATCH="${DEVICE_EVAL_BATCH_SIZE:-4}"

if [[ $# -ge 1 ]]; then
  CKPT="$1"
elif [[ -n "${STEP:-}" && -n "${CKPT_ROOT:-}" ]]; then
  CKPT="${CKPT_ROOT}/step${STEP}"
else
  echo "Usage: $0 <checkpoint_dir>   OR   STEP=N CKPT_ROOT=... $0" >&2
  exit 1
fi

STEP_NUM="$(basename "$CKPT" | sed -E 's/^step([0-9]+)$/\1/')"
OUT="${OUT_DIR}/step${STEP_NUM}_task_loss.json"
mkdir -p "$OUT_DIR"

echo "[$(date -Is)] eval $CKPT → $OUT"
python "$EVAL_PY" \
  --checkpoint "$CKPT" \
  --out "$OUT" \
  --run-name "$RUN_NAME" \
  --device-eval-batch-size "$BATCH"
