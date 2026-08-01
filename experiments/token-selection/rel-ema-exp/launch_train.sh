#!/usr/bin/env bash
# Hardware-agnostic launch for REL no-init exp-α (rel-ema-exp).
# Near-clone of rel-ema-refhq/launch_train.sh — only EMA seed + α schedule differ
# (wired via YAML). This arm must NOT set REF_PT / ema.seed_mode=refhq.
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
# Discovers world size from NUM_GPUS / CUDA_VISIBLE_DEVICES / nvidia-smi.
# Does not hardcode device IDs, node names, or GPU count.
#
# Usage (from anywhere):
#   export EDULLM_ROOT=/path/to/edullm
#   export OLMO_CORE_DIR=/path/to/OLMo-core   # pinned revision in YAML (or let prepare clone)
#   export RUN_DIR=/path/to/empty/scratch
#   # Optional: CUDA_VISIBLE_DEVICES=0,1  or  NUM_GPUS=2
#   bash $EDULLM_ROOT/experiments/token-selection/rel-ema-exp/launch_train.sh prepare
#   bash $EDULLM_ROOT/experiments/token-selection/rel-ema-exp/launch_train.sh train
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
CFG_REL="${CFG_REL:-rel-ema-exp/configs/run_rel_ema_exp_10b.yaml}"
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
NUM_WORKERS="${NUM_WORKERS:-}"
RDZV_PORT="${RDZV_PORT:-29531}"
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

mkdir -p "$WORK/logs" "$WORK/progress" "$WORK/task_loss_results/rel-ema-exp"
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

# Rewrite runtime YAML paths into RUN_DIR (output_dir + task_loss results absolute).
RUNTIME_CFG="$WORK/run_rel_ema_exp.runtime.yaml"
RANK_MICROBATCH_SIZE="${RANK_MICROBATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
NUM_GPUS="${NUM_GPUS}" CFG="${CFG}" RUNTIME_CFG="${RUNTIME_CFG}" WORK="${WORK}" \
python - <<'PY'
import os
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(os.environ["CFG"]).read_text(encoding="utf-8"))
work = Path(os.environ["WORK"])
cfg["output_dir"] = str(work / "data" / "rel_ema_exp_10b")
train = cfg.setdefault("train", {})
train["num_gpus"] = int(os.environ["NUM_GPUS"])
train["cuda_visible_devices"] = ""
rms = os.environ.get("RANK_MICROBATCH_SIZE", "").strip()
if rms:
    train["rank_microbatch_size"] = int(rms)
nw = os.environ.get("NUM_WORKERS", "").strip()
if nw:
    train["num_workers"] = int(nw)
eval_cfg = cfg.setdefault("eval", {})
tl = eval_cfg.setdefault("task_loss", {})
tl["enabled"] = True
tl["results_dir"] = str(work / "task_loss_results" / "rel-ema-exp")
# Fail closed: never silently seed EMA from RefHQ on this arm.
ema = cfg.setdefault("ema", {})
ema["seed_mode"] = "zero"
ema["schedule"] = str(ema.get("schedule") or cfg.get("alpha_schedule") or "exp")
if "tau" not in ema:
    ema["tau"] = float(cfg.get("alpha_tau", 300))
ref = cfg.setdefault("reference", {})
ref["load_path"] = None
Path(os.environ["RUNTIME_CFG"]).write_text(
    yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
)
print("wrote", os.environ["RUNTIME_CFG"])
PY

OUT_DIR="$WORK/data/rel_ema_exp_10b"
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
  # Fail closed: this arm must stay zero-init EMA + exp α (not RefHQ-seeded).
  python - <<'PY' "$RUNTIME_CFG"
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
ema = cfg.get("ema") or {}
seed = str(ema.get("seed_mode") or cfg.get("ema_seed_mode") or "zero").lower()
sched = str(ema.get("schedule") or cfg.get("alpha_schedule") or "linear").lower()
tau = float(ema.get("tau") if ema.get("tau") is not None else cfg.get("alpha_tau", 300))
if seed != "zero":
    raise SystemExit(f"rel-ema-exp requires ema.seed_mode=zero, got {seed!r}")
if sched != "exp":
    raise SystemExit(f"rel-ema-exp requires alpha schedule=exp, got {sched!r}")
if abs(tau - 300.0) > 1e-9:
    raise SystemExit(f"rel-ema-exp requires alpha tau=300, got {tau}")
if int(cfg.get("t0_steps", -1)) != 0:
    raise SystemExit(f"rel-ema-exp requires t0_steps=0, got {cfg.get('t0_steps')!r}")
rid = str(cfg.get("run_id") or "")
if rid == "rel-ema-10b-scratch-v1" or not rid.startswith("rel-ema-exp-"):
    raise SystemExit(
        f"rel-ema-exp requires a new run_id (not rel-ema-10b-scratch-v1); got {rid!r}"
    )
print(f"independent vars ok: seed_mode={seed} schedule={sched} tau={tau} t0=0 run_id={rid}")
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
  _REL_EMA_EXP_RUN_NAME="${WANDB_RUN_NAME:-$(python -c "import yaml; from pathlib import Path; print(yaml.safe_load(Path('${RUNTIME_CFG}').read_text(encoding='utf-8'))['run_id'])")}"
  # shellcheck disable=SC1091
  source "${TS_ROOT}/token_selection/scripts/wandb_env.sh" "rel-ema-exp" "${_REL_EMA_EXP_RUN_NAME}"
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
