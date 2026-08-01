#!/usr/bin/env bash
# Hardware-agnostic launch for REL RefHQ-seeded constant-α (rel-ema-refhq).
# Near-clone of rel-ema-exp/launch_train.sh — only EMA seed + α schedule differ
# (wired via YAML).
#
# Ephemeral empty-scratch contract:
#   - Set RUN_DIR (or WORK) to a job scratch dir; do not assume prior contents.
#   - Stage train shards: YAML data.dataset_id → edullm_data.read → s3://edullm-data/
#   - Never uses s3://edullm-datasets/. Do not assume laptop/FarmShare corpora or venvs.
#   - Artifacts remain on scratch and upload to W&B.
#   - --resume restores WANDB_RESUME_ARTIFACT when save_folder is empty.
#
# W&B project token-selection is the artifact store. Push
# wandb-session.env via scripts/farmshare/push_wandb_session_to_farmshare.sh
# "$RUN_DIR" (or set WANDB_SESSION_ENV). Local smoke: WANDB_MODE=disabled.
#
# REF_PT is optional: if unset, --launch auto-materializes reference.s3_uri
# (RefHQ step1315 DistCP on S3) into TOKEN_SELECTION_REF_CACHE.
#
# Discovers world size from NUM_GPUS / CUDA_VISIBLE_DEVICES / nvidia-smi.
# Does not hardcode device IDs, node names, or GPU count.
#
# Usage (from anywhere):
#   export EDULLM_ROOT=/path/to/edullm
#   export OLMO_CORE_DIR=/path/to/OLMo-core   # pinned revision in YAML (or let prepare clone)
#   export RUN_DIR=/path/to/empty/scratch
#   # Optional: REF_PT=/path/to/refhq_step1315_model.pt
#   # Optional: CUDA_VISIBLE_DEVICES=0,1  or  NUM_GPUS=2
#   bash $EDULLM_ROOT/experiments/token-selection/rel-ema-refhq/launch_train.sh prepare
#   bash $EDULLM_ROOT/experiments/token-selection/rel-ema-refhq/launch_train.sh train
#
# Resume (durable):
#   bash .../launch_train.sh train --resume
set -euo pipefail

MODE="${1:-train}"
shift || true
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"          # experiments/token-selection
EDULLM_ROOT="${EDULLM_ROOT:-$(cd "$TS_ROOT/../.." && pwd)}"
CFG_REL="${CFG_REL:-rel-ema-refhq/configs/run_rel_ema_refhq_10b.yaml}"
CFG="$TS_ROOT/$CFG_REL"
METHOD="${METHOD:-rel_ema}"

if [[ -z "${RUN_DIR:-${WORK:-}}" ]]; then
  echo "ERROR: set RUN_DIR (or WORK) to an empty job scratch directory." >&2
  echo "  Scratch is ephemeral — stage edullm-data each job; durable ckpts live on S3." >&2
  exit 2
fi
RUN_DIR="${RUN_DIR:-$WORK}"
WORK="${WORK:-$RUN_DIR}"
OLMO_CORE_DIR="${OLMO_CORE_DIR:-${OLMO_ROOT:-$WORK/OLMo-core}}"
OLMO_CORE_REVISION="${OLMO_CORE_REVISION:-99e0009ed67679c90da970ec5ba439c9459e3757}"
RANK_MICROBATCH_SIZE="${RANK_MICROBATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RDZV_PORT="${RDZV_PORT:-29532}"
REF_PT="${REF_PT:-}"
OFFLINE="${OFFLINE:-0}"

if [[ -n "${TRAIN_DATA_S3:-}" ]]; then
  echo "ERROR: TRAIN_DATA_S3 is no longer supported (got ${TRAIN_DATA_S3})." >&2
  echo "  Set data.dataset_id in the YAML (pretrain/regmix-10b); staging uses edullm-data." >&2
  exit 2
fi

log() { echo "[$(date -Is)] $*"; }

discover_num_gpus() {
  if [[ -n "${NUM_GPUS:-}" ]]; then
    echo "$NUM_GPUS"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local csv="${CUDA_VISIBLE_DEVICES// /}"
    if [[ -n "$csv" ]]; then
      echo "$csv" | awk -F',' '{print NF}'
      return
    fi
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
    return
  fi
  echo 1
}

NUM_GPUS="$(discover_num_gpus)"
if [[ -z "$NUM_GPUS" || "$NUM_GPUS" -lt 1 ]]; then
  NUM_GPUS=1
fi
export TASK_LOSS_STRICT=1
export TASK_LOSS_NPROC="${TASK_LOSS_NPROC:-${NUM_GPUS}}"

mkdir -p "$WORK/logs" "$WORK/progress"
export PYTHONPATH="${TS_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export TOKEN_SELECTION_SKIP_IDLE_CHECK="${TOKEN_SELECTION_SKIP_IDLE_CHECK:-1}"

# Optional convenience only — never required. Prefer the active interpreter on PATH.
if [[ -x "$WORK/venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$WORK/venv/bin/activate"
elif [[ -x "$EDULLM_ROOT/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$EDULLM_ROOT/.venv/bin/activate"
fi

# Rewrite runtime YAML paths into RUN_DIR (output_dir absolute); inject REF_PT.
RUNTIME_CFG="$WORK/run_rel_ema_refhq.runtime.yaml"
REF_PT="${REF_PT}" RANK_MICROBATCH_SIZE="${RANK_MICROBATCH_SIZE}" \
NUM_GPUS="${NUM_GPUS}" NUM_WORKERS="${NUM_WORKERS}" \
CFG="${CFG}" RUNTIME_CFG="${RUNTIME_CFG}" WORK="${WORK}" python - <<'PY'
import os
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(os.environ["CFG"]).read_text(encoding="utf-8"))
cfg["output_dir"] = str(Path(os.environ["WORK"]) / "data" / "rel_ema_refhq_10b")
train = cfg.setdefault("train", {})
train["num_gpus"] = int(os.environ["NUM_GPUS"])
train["cuda_visible_devices"] = ""
rms = os.environ.get("RANK_MICROBATCH_SIZE", "").strip()
if rms:
    train["rank_microbatch_size"] = int(rms)
train["num_workers"] = int(os.environ["NUM_WORKERS"])
ref_pt = os.environ.get("REF_PT", "").strip()
if ref_pt:
    cfg.setdefault("reference", {})["load_path"] = ref_pt
Path(os.environ["RUNTIME_CFG"]).write_text(
    yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
)
print("wrote", os.environ["RUNTIME_CFG"])
PY

OUT_DIR="$WORK/data/rel_ema_refhq_10b"
TOK_DIR="$OUT_DIR/tokens"

prepare() {
  log "prepare: stage train tokens from edullm-data (data.dataset_id) into empty scratch"
  mkdir -p "$TOK_DIR"
  if [[ "$OFFLINE" == "1" ]]; then
    if [[ ! -f "$TOK_DIR/manifest.json" ]]; then
      echo "ERROR: OFFLINE=1 but missing staged tokens at $TOK_DIR/manifest.json" >&2
      echo "  On ephemeral hosts leave OFFLINE unset and stage from edullm-data." >&2
      exit 2
    fi
    log "OFFLINE=1 — reusing staged $TOK_DIR (same-job only; not durable across wipe)"
    python - "$RUNTIME_CFG" "$OUT_DIR" <<'PY'
import sys
from pathlib import Path
from token_selection.scripts import load_config
from token_selection.scripts.edullm_data_tokens import ensure_order_contract

ensure_order_contract(load_config(Path(sys.argv[1])), Path(sys.argv[2]))
print("order contract ok")
PY
  else
    python - "$RUNTIME_CFG" "$OUT_DIR" <<'PY'
import sys
from pathlib import Path
from token_selection.scripts import load_config
from token_selection.scripts.edullm_data_tokens import (
    ensure_order_contract,
    ensure_train_tokens,
)

cfg = load_config(Path(sys.argv[1]))
out = Path(sys.argv[2])
if not (cfg.get("data") or {}).get("dataset_id"):
    raise SystemExit("data.dataset_id required (e.g. pretrain/regmix-10b)")
manifest = ensure_train_tokens(cfg, out / "tokens")
ensure_order_contract(cfg, out)
print(
    f"staged {manifest.get('dataset_id')}/{manifest.get('dataset_version')} "
    f"n_tokens={manifest.get('n_tokens')} → {out / 'tokens'}"
)
PY
  fi
  if [[ ! -d "$OLMO_CORE_DIR/.git" && ! -d "$OLMO_CORE_DIR/src/olmo_core" ]]; then
    log "clone OLMo-core → $OLMO_CORE_DIR (job-local; not a persistent host assumption)"
    git clone https://github.com/edu-llm/OLMo-core "$OLMO_CORE_DIR"
  fi
  if [[ -d "$OLMO_CORE_DIR/.git" ]]; then
    git -C "$OLMO_CORE_DIR" fetch --all --tags || true
    git -C "$OLMO_CORE_DIR" checkout "$OLMO_CORE_REVISION"
  fi
  log "prepare done (NUM_GPUS=$NUM_GPUS)"
}

train() {
  local resume_flag=()
  if [[ " ${EXTRA_ARGS[*]} " == *" --resume "* ]]; then
    resume_flag=(--resume)
    log "resume: spine restores the run's W&B checkpoint artifact if needed"
  fi
  if [[ -n "${WANDB_RESUME_ARTIFACT:-}" ]]; then
    resume_flag+=(--wandb-resume-artifact "$WANDB_RESUME_ARTIFACT")
  fi
  # Fail closed: local load_path or S3 provenance (auto-materialize at --launch).
  python - <<'PY' "$RUNTIME_CFG"
import sys, yaml
from pathlib import Path
from token_selection.olmo_ext.refhq_materialize import reference_source_ok
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not reference_source_ok(cfg, method="rel_ema"):
    raise SystemExit(
        "rel-ema-refhq needs reference.load_path or reference.s3_uri "
        "(YAML already has s3_uri; set REF_PT to override with a local .pt)."
    )
ref = (cfg.get("reference") or {}).get("load_path")
s3 = (cfg.get("reference") or {}).get("s3_uri")
print(f"EMA seed source ok: load_path={ref!r} s3_uri={s3!r}")
PY
  log "train: torchrun nproc=$NUM_GPUS method=$METHOD cfg=$RUNTIME_CFG"
  # GBS must divide evenly by world_size * rank_microbatch.
  python - <<'PY' "$RUNTIME_CFG" "$NUM_GPUS"
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
gbs = int(cfg["train"]["global_batch_size"])
rmbs = int(cfg["train"]["rank_microbatch_size"])
ws = int(sys.argv[2])
if gbs % (ws * rmbs) != 0:
    raise SystemExit(
        f"global_batch_size={gbs} not divisible by world_size*rank_microbatch "
        f"({ws}*{rmbs}={ws*rmbs}). Set RANK_MICROBATCH_SIZE so it divides."
    )
print(f"batch ok: gbs={gbs} ws={ws} rmbs={rmbs} accum={gbs // (ws * rmbs)}")
PY
  _REL_EMA_REFHQ_RUN_NAME="${WANDB_RUN_NAME:-$(python -c "import yaml; from pathlib import Path; print(yaml.safe_load(Path('${RUNTIME_CFG}').read_text(encoding='utf-8'))['run_id'])")}"
  # shellcheck disable=SC1091
  source "${TS_ROOT}/token_selection/scripts/wandb_env.sh" "rel-ema-refhq" "${_REL_EMA_REFHQ_RUN_NAME}"
  torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    --rdzv_endpoint="localhost:${RDZV_PORT}" \
    -m token_selection.scripts.train_olmo_template \
    --config "$RUNTIME_CFG" \
    --method "$METHOD" \
    --olmo-root "$OLMO_CORE_DIR" \
    --launch \
    "${resume_flag[@]}"
}

case "$MODE" in
  prepare) prepare ;;
  train) train ;;
  all) prepare; EXTRA_ARGS=(); train ;;
  *)
    echo "Usage: $0 {prepare|train|all} [--resume]" >&2
    exit 2
    ;;
esac
