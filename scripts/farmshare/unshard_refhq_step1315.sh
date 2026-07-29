#!/usr/bin/env bash
# Unshard RefHQ 5.5B reference checkpoint (step1315) on FarmShare.
# Usage (login node):
#   export RUN_DIR=/scratch/users/nzhao2/agent-runs/refhq-unshard-$(date -u +%Y%m%dT%H%M%SZ)
#   bash scripts/farmshare/unshard_refhq_step1315.sh
# Or submit:
#   sbatch --exclude=wheat-01 scripts/farmshare/unshard_refhq_step1315.sbatch
set -Eeuo pipefail

RUN_DIR="${RUN_DIR:-/scratch/users/nzhao2/agent-runs/refhq-unshard}"
EDULLM_ROOT="${EDULLM_ROOT:-$HOME/edullm}"
TS_ROOT="${TS_ROOT:-$EDULLM_ROOT/experiments/token-selection}"
EXPORT_PY="${EXPORT_PY:-$TS_ROOT/reference/export_refhq_reference.py}"
REF_CKPT_S3="${REF_CKPT_S3:-s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/step1315/}"

REF_DIR="${REF_DIR:-$RUN_DIR/reference}"
REF_PT="${REF_PT:-$REF_DIR/refhq_step1315_model.pt}"
WORK_DIR="${WORK_DIR:-$REF_DIR/work}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"

# Reuse a known-good venv with olmo_core if the run dir has none yet.
FALLBACK_VENV="${FALLBACK_VENV:-/scratch/users/nzhao2/agent-runs/rho-excess-10b-l40s/venv}"
LADDER_VENV="${LADDER_VENV:-/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/venv}"
OLMO_CORE_DIR="${OLMO_CORE_DIR:-$RUN_DIR/OLMo-core}"
OLMO_CORE_REVISION="${OLMO_CORE_REVISION:-99e0009ed67679c90da970ec5ba439c9459e3757}"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$REF_DIR"

log() { echo "[$(date -Is)] $*"; }

export PATH="$HOME/.local/bin:$HOME/tools/aws/bin:$PATH"

if [[ -x "$LADDER_VENV/bin/python" ]]; then
  PYTHON="$LADDER_VENV/bin/python"
elif [[ -x "$RUN_DIR/venv/bin/python" ]]; then
  PYTHON="$RUN_DIR/venv/bin/python"
elif [[ -x "$FALLBACK_VENV/bin/python" ]]; then
  PYTHON="$FALLBACK_VENV/bin/python"
else
  PYTHON="$(command -v python3)"
fi

ensure_olmo_core() {
  if "$PYTHON" -c "from olmo_core.distributed.checkpoint import unshard_checkpoint" 2>/dev/null; then
    return 0
  fi
  log "Installing pinned OLMo-core @ ${OLMO_CORE_REVISION}"
  if [[ ! -d "$OLMO_CORE_DIR/.git" ]]; then
    git clone https://github.com/edu-llm/OLMo-core "$OLMO_CORE_DIR" \
      || git clone https://github.com/allenai/OLMo-core "$OLMO_CORE_DIR"
  fi
  git -C "$OLMO_CORE_DIR" fetch --all --tags || true
  git -C "$OLMO_CORE_DIR" checkout "$OLMO_CORE_REVISION" || true
  "$PYTHON" -m pip install -q -e "$OLMO_CORE_DIR"
  "$PYTHON" -c "from olmo_core.distributed.checkpoint import unshard_checkpoint"
}

AWS_ENV="${AWS_SESSION_ENV:-}"
for cand in \
  "$RUN_DIR/aws-session.env" \
  "/scratch/users/nzhao2/agent-runs/aws-session.env" \
  "/scratch/users/nzhao2/agent-runs/rho-excess-10b-l40s/aws-session.env"
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

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI not found; run sb-aws-creds login on FarmShare first" >&2
  exit 1
fi

aws sts get-caller-identity --output text >/dev/null

if [[ ! -f "$EXPORT_PY" ]]; then
  # Fall back to the rho staging tree if the home repo is absent.
  ALT="/scratch/users/nzhao2/agent-runs/rho-excess-10b-l40s/edullm/experiments/token-selection/reference/export_refhq_reference.py"
  if [[ -f "$ALT" ]]; then
    EXPORT_PY="$ALT"
  else
    echo "ERROR: missing export script at $EXPORT_PY" >&2
    exit 2
  fi
fi

"$PYTHON" -c "import olmo_core" || {
  echo "ERROR: olmo_core missing in $PYTHON" >&2
  exit 1
}
ensure_olmo_core

log "RUN_DIR=$RUN_DIR"
log "REF_PT=$REF_PT"
log "S3=$REF_CKPT_S3"

if [[ -f "$REF_PT" && "${FORCE:-0}" != "1" ]]; then
  log "SKIP: $REF_PT already exists (set FORCE=1 to redo)"
  ls -lh "$REF_PT" "${REF_PT%.pt}.json"
  exit 0
fi

"$PYTHON" "$EXPORT_PY" \
  --s3-uri "$REF_CKPT_S3" \
  --work-dir "$WORK_DIR" \
  --output "$REF_PT" \
  2>&1 | tee "$LOG_DIR/export_refhq.log"

test -f "$REF_PT"
ls -lh "$REF_PT" "${REF_PT%.pt}.json"
log "READY reference.load_path=$REF_PT"
