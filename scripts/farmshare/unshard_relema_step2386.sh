#!/usr/bin/env bash
# Unshard REL-EMA final distcp checkpoint (step2386) to a flat model.pt for inference.
#
# Uses the already-synced FarmShare copy unde
#   /scratch/users/nzhao2/checkpoints/token-selection-370m/rel-ema/step2386
# (no S3 sync required).
set -Eeuo pipefail

: "${RUN_DIR:?}"

LADDER_VENV="${LADDER_VENV:-/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/venv}"
CKPT_DIR="${CKPT_DIR:-/scratch/users/nzhao2/checkpoints/token-selection-370m/rel-ema/step2386}"
OUT_PT="${OUT_PT:-$RUN_DIR/relema_step2386_model.pt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNSHARD_PY="${UNSHARD_PY:-$SCRIPT_DIR/unshard_distcp_to_model_pt.py}"

mkdir -p "$RUN_DIR/logs" "$(dirname "$OUT_PT")"

if [[ -x "$LADDER_VENV/bin/python" ]]; then
  PYTHON="$LADDER_VENV/bin/python"
else
  PYTHON="$(command -v python3)"
fi

if [[ ! -f "$CKPT_DIR/model_and_optim/.metadata" ]]; then
  echo "ERROR: missing distcp metadata under $CKPT_DIR/model_and_optim" >&2
  exit 2
fi
if [[ ! -f "$UNSHARD_PY" ]]; then
  echo "ERROR: missing $UNSHARD_PY" >&2
  exit 2
fi

if [[ -f "$OUT_PT" && "${FORCE:-0}" != "1" ]]; then
  echo "SKIP unshard (exists): $OUT_PT"
else
  echo "Unsharding $CKPT_DIR -> $OUT_PT"
  "$PYTHON" "$UNSHARD_PY" \
    --checkpoint-dir "$CKPT_DIR" \
    --output "$OUT_PT" \
    --step 2386 \
    --work-dir "$RUN_DIR/work_unshard"
fi

ls -lh "$OUT_PT"
"$PYTHON" - <<PY
import torch
from pathlib import Path
p = Path("${OUT_PT}")
obj = torch.load(p, map_location="cpu", weights_only=False)
if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
    sd = obj["model"]
elif isinstance(obj, dict):
    sd = {k: v for k, v in obj.items() if torch.is_tensor(v)}
else:
    raise SystemExit(f"unexpected type {type(obj)}")
emb = sd.get("embeddings.weight")
print("n_tensors", len(sd))
print("embeddings.weight", None if emb is None else tuple(emb.shape))
if emb is None or tuple(emb.shape) != (100352, 1024):
    raise SystemExit("bad embedding shape; aborting before smoke test")
print("SHAPE_OK")
PY
