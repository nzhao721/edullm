#!/usr/bin/env bash
# Sync + unshard all (or selected) olmo-core distcp checkpoints for one run.
#
# Required:
#   RUN_DIR          scratch working directory
#   RUN_LABEL        short name (e.g. ce-regmix10b, blade-regmix10b)
#   S3_CKPT_PREFIX   s3://.../checkpoints/  (trailing slash ok)
#
# Optional:
#   STEPS            space-separated step numbers (default: discover step* on S3)
#   FORCE            1 to redo existing model.pt outputs
set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${RUN_LABEL:?}"
# S3_CKPT_PREFIX required unless SKIP_S3_SYNC=1 (local DISTCP_ROOT only).
SKIP_S3_SYNC="${SKIP_S3_SYNC:-0}"
if [[ "$SKIP_S3_SYNC" != "1" ]]; then
  : "${S3_CKPT_PREFIX:?}"
else
  S3_CKPT_PREFIX="${S3_CKPT_PREFIX:-local}"
fi

LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"
DISTCP_ROOT="${DISTCP_ROOT:-$RUN_DIR/distcp}"
OUT_ROOT="${OUT_ROOT:-$RUN_DIR/unsharded}"
LADDER_VENV="${LADDER_VENV:-/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/venv}"
OLMO_CORE_DIR="${OLMO_CORE_DIR:-$RUN_DIR/OLMo-core}"
OLMO_CORE_REVISION="${OLMO_CORE_REVISION:-99e0009ed67679c90da970ec5ba439c9459e3757}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNSHARD_PY="${UNSHARD_PY:-$SCRIPT_DIR/unshard_distcp_to_model_pt.py}"

mkdir -p "$LOG_DIR" "$DISTCP_ROOT" "$OUT_ROOT"

log() { echo "[$(date -Is)] [$RUN_LABEL] $*"; }

export PATH="$HOME/.local/bin:$HOME/tools/aws/bin:$PATH"
if [[ -x "$LADDER_VENV/bin/python" ]]; then
  PYTHON="$LADDER_VENV/bin/python"
else
  PYTHON="$(command -v python3)"
fi

ensure_olmo_core() {
  if "$PYTHON" -c "from olmo_core.distributed.checkpoint import unshard_checkpoint" 2>/dev/null; then
    return 0
  fi
  log "Installing OLMo-core @ ${OLMO_CORE_REVISION}"
  if [[ ! -d "$OLMO_CORE_DIR/.git" ]]; then
    git clone https://github.com/edu-llm/OLMo-core "$OLMO_CORE_DIR" \
      || git clone https://github.com/allenai/OLMo-core "$OLMO_CORE_DIR"
  fi
  git -C "$OLMO_CORE_DIR" fetch --all --tags || true
  git -C "$OLMO_CORE_DIR" checkout "$OLMO_CORE_REVISION" || true
  "$PYTHON" -m pip install -q -e "$OLMO_CORE_DIR"
}

if [[ "$SKIP_S3_SYNC" != "1" ]]; then
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
fi

ensure_olmo_core

if [[ ! -f "$UNSHARD_PY" ]]; then
  echo "ERROR: missing $UNSHARD_PY" >&2
  exit 2
fi

S3_CKPT_PREFIX="${S3_CKPT_PREFIX%/}/"

if [[ -z "${STEPS:-}" ]]; then
  if [[ "$SKIP_S3_SYNC" == "1" ]]; then
    mapfile -t STEP_LIST < <(
      find "$DISTCP_ROOT" -maxdepth 1 -type d -name 'step*' \
        | sed 's#.*/step##' \
        | sed 's/-.*//' \
        | sort -n -u
    )
  else
    mapfile -t STEP_LIST < <(
      aws s3 ls "$S3_CKPT_PREFIX" \
        | awk '{print $2}' \
        | sed 's#/##' \
        | grep '^step[0-9]' \
        | sed 's/^step//' \
        | sed 's/-.*//' \
        | sort -n -u
    )
  fi
else
  read -r -a STEP_LIST <<< "$STEPS"
fi

if [[ ${#STEP_LIST[@]} -eq 0 ]]; then
  echo "ERROR: no steps found (S3=$S3_CKPT_PREFIX DISTCP=$DISTCP_ROOT)" >&2
  exit 2
fi

log "steps=${STEP_LIST[*]}"
if [[ "$SKIP_S3_SYNC" == "1" ]]; then
  log "local-only DISTCP_ROOT=$DISTCP_ROOT (SKIP_S3_SYNC=1)"
else
  log "syncing distcp from $S3_CKPT_PREFIX -> $DISTCP_ROOT"
fi

failed=0
for step in "${STEP_LIST[@]}"; do
  out_pt="$OUT_ROOT/step${step}/model.pt"
  if [[ -f "$out_pt" && "${FORCE:-0}" != "1" ]]; then
    log "SKIP step${step} (exists)"
    continue
  fi
  local_ckpt="$DISTCP_ROOT/step${step}"
  if [[ ! -f "$local_ckpt/model_and_optim/.metadata" ]]; then
    if [[ "$SKIP_S3_SYNC" == "1" ]]; then
      log "FAIL step${step}: missing local distcp metadata under $local_ckpt"
      failed=$((failed + 1))
      continue
    fi
    log "sync step${step}"
    mkdir -p "$local_ckpt"
    aws s3 sync "${S3_CKPT_PREFIX}step${step}/" "$local_ckpt/" --only-show-errors
  fi
  if [[ ! -f "$local_ckpt/model_and_optim/.metadata" ]]; then
    log "FAIL step${step}: missing distcp metadata after sync"
    failed=$((failed + 1))
    continue
  fi
  log "unshard step${step} -> $out_pt"
  if ! "$PYTHON" "$UNSHARD_PY" \
    --checkpoint-dir "$local_ckpt" \
    --output "$out_pt" \
    --step "$step" \
    2>&1 | tee -a "$LOG_DIR/unshard_${RUN_LABEL}_step${step}.log"; then
    log "FAIL step${step} unshard"
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
    step = int(p.parent.name.replace("step", "").split("-")[0])
    rows.append({"step": step, "model_pt": str(p), "bytes": p.stat().st_size})
Path("${manifest}").write_text(json.dumps({
    "run_label": "${RUN_LABEL}",
    "s3_prefix": "${S3_CKPT_PREFIX}",
    "distcp_root": "${DISTCP_ROOT}",
    "checkpoints": rows,
}, indent=2) + "\\n")
print(f"wrote {len(rows)} entries -> {out / 'manifest.json'}")
PY

log "done $RUN_LABEL failures=$failed; outputs under $OUT_ROOT"
find "$OUT_ROOT" -name 'model.pt' | sort
exit "$failed"
