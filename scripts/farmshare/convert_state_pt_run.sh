#!/usr/bin/env bash
# Sync CE/BLADE state.pt checkpoints from S3 and convert to inference-ready model.pt.
#
# Required:
#   RUN_DIR, RUN_LABEL, S3_CKPT_PREFIX
# Optional:
#   STEPS (default: 250 500 750 1000 1250 1500 1750 2000 2250 2384)
#   FORCE=1 to redo existing outputs
set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${RUN_LABEL:?}"
: "${S3_CKPT_PREFIX:?}"

LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"
CKPT_ROOT="${CKPT_ROOT:-$RUN_DIR/checkpoints}"
OUT_ROOT="${OUT_ROOT:-$RUN_DIR/models}"
LADDER_VENV="${LADDER_VENV:-/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/venv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERT_PY="${CONVERT_PY:-$SCRIPT_DIR/convert_state_pt.py}"

mkdir -p "$LOG_DIR" "$CKPT_ROOT" "$OUT_ROOT"

log() { echo "[$(date -Is)] [$RUN_LABEL] $*"; }

export PATH="$HOME/.local/bin:$HOME/tools/aws/bin:$PATH"
if [[ -x "$LADDER_VENV/bin/python" ]]; then
  PYTHON="$LADDER_VENV/bin/python"
else
  PYTHON="$(command -v python3)"
fi

AWS_ENV="${AWS_SESSION_ENV:-}"
for cand in \
  "$RUN_DIR/aws-session.env" \
  "/scratch/users/nzhao2/agent-runs/aws-session.env" \
  "/scratch/users/nzhao2/agent-runs/rho-excess-10b-l40s/aws-session.env" \
  "/scratch/users/nzhao2/agent-runs/refhq-unshard-20260727/aws-session.env"
do
  if [[ -z "$AWS_ENV" && -f "$cand" ]]; then
    AWS_ENV="$cand"
  fi
done
if [[ -n "$AWS_ENV" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$AWS_ENV"
  set -u
  unset AWS_PROFILE
fi

command -v aws >/dev/null
aws sts get-caller-identity --output text >/dev/null

if [[ ! -f "$CONVERT_PY" ]]; then
  echo "ERROR: missing $CONVERT_PY" >&2
  exit 2
fi

S3_CKPT_PREFIX="${S3_CKPT_PREFIX%/}/"
STEPS="${STEPS:-250 500 750 1000 1250 1500 1750 2000 2250 2384}"
read -r -a STEP_LIST <<< "$STEPS"

log "steps=${STEP_LIST[*]}"
log "S3=$S3_CKPT_PREFIX"
log "OUT=$OUT_ROOT"

failed=0
for step in "${STEP_LIST[@]}"; do
  out_pt="$OUT_ROOT/step${step}/model.pt"
  if [[ -f "$out_pt" && "${FORCE:-0}" != "1" ]]; then
    log "SKIP step${step} (exists)"
    continue
  fi
  local_ckpt="$CKPT_ROOT/step${step}"
  if [[ ! -f "$local_ckpt/state.pt" ]]; then
    log "sync step${step}"
    mkdir -p "$local_ckpt"
    aws s3 sync "${S3_CKPT_PREFIX}step${step}/" "$local_ckpt/" --only-show-errors
  fi
  if [[ ! -f "$local_ckpt/state.pt" ]]; then
    log "FAIL step${step}: missing state.pt after sync"
    failed=$((failed + 1))
    continue
  fi
  log "convert step${step} -> $out_pt"
  if ! "$PYTHON" "$CONVERT_PY" \
    --checkpoint "$local_ckpt" \
    --out "$out_pt" \
    2>&1 | tee "$LOG_DIR/convert_${RUN_LABEL}_step${step}.log"; then
    log "FAIL step${step} convert"
    failed=$((failed + 1))
    continue
  fi
  ls -lh "$out_pt"
done

manifest="$OUT_ROOT/manifest.json"
"$PYTHON" - <<PY
import json
from pathlib import Path
out = Path("${OUT_ROOT}")
rows = []
for p in sorted(out.glob("step*/model.pt")):
    step = int(p.parent.name.replace("step", ""))
    rows.append({"step": step, "model_pt": str(p), "bytes": p.stat().st_size})
Path("${manifest}").write_text(json.dumps({
    "run_label": "${RUN_LABEL}",
    "s3_prefix": "${S3_CKPT_PREFIX}",
    "checkpoints": rows,
}, indent=2) + "\\n")
print(f"wrote {len(rows)} entries -> {out / 'manifest.json'}")
PY

log "done $RUN_LABEL failures=$failed"
find "$OUT_ROOT" -name 'model.pt' | sort
exit "$failed"
