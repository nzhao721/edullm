#!/usr/bin/env bash
# RHO-1 FarmShare / local helper — RegMix 10B (run_id=rho-1-regmix10b-v1).
#
# Does NOT resume the discarded rho-excess-10b-scratch-v1 (~step200) run.
# World size / GPU count are discovered from the environment (no hard-coded host).
#
# Usage:
#   export RUN_DIR=/scratch/users/$USER/agent-runs/rho-1-regmix10b-v1
#   export EDULLM_ROOT=/path/to/edullm
#   bash $EDULLM_ROOT/experiments/token-selection/rho-1/farmshare/run_rho_train.sh prepare
#   bash $EDULLM_ROOT/experiments/token-selection/rho-1/farmshare/run_rho_train.sh train
#
# Crash resume within THIS run_id only:
#   RESUME=1 bash .../run_rho_train.sh train
set -euo pipefail

MODE="${1:-prepare}"  # prepare | train | all

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"          # experiments/token-selection
EDULLM_ROOT="${EDULLM_ROOT:-$(cd "$TS_ROOT/../.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$TS_ROOT}"                 # PYTHONPATH root for token_selection
RUN_DIR="${RUN_DIR:-${PWD}/rho-1-regmix10b-v1}"
WORK="${WORK:-$RUN_DIR}"

# Discover GPU count: explicit NUM_GPUS > Slurm > CUDA_VISIBLE_DEVICES length > 1
_discover_num_gpus() {
  if [[ -n "${NUM_GPUS:-}" ]]; then
    echo "$NUM_GPUS"
    return
  fi
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    echo "${SLURM_GPUS_ON_NODE%%(*}"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local IFS=,
    # shellcheck disable=SC2206
    local arr=(${CUDA_VISIBLE_DEVICES})
    echo "${#arr[@]}"
    return
  fi
  echo 1
}

NUM_GPUS="$(_discover_num_gpus)"
RANK_MICROBATCH_SIZE="${RANK_MICROBATCH_SIZE:-65536}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DP_TYPE="${DP_TYPE:-hsdp}"
ATTN_BACKEND="${ATTN_BACKEND:-auto}"
# Match RefHQ / YAML contract (compile_model=true). Override with COMPILE_MODEL=0 if needed.
COMPILE_MODEL="${COMPILE_MODEL:-1}"
RESUME="${RESUME:-0}"
if [[ "$RESUME" == "1" ]]; then
  FROM_SCRATCH=0
else
  FROM_SCRATCH="${FROM_SCRATCH:-1}"
fi

RUN_ID="rho-1-regmix10b-v1"
DISCARDED_RUN_ID="rho-excess-10b-scratch-v1"
CKPT_S3="${CKPT_S3:-s3://edullm-checkpoints/token-sel/rho-1}"
TRAIN_DATA_S3="${TRAIN_DATA_S3:-s3://edullm-datasets/regmix/regmix-10b/tokenized}"
REF_CKPT_S3="${REF_CKPT_S3:-s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/step1315/}"
# Experiment results (metrics) live under the same arm prefix on edullm-checkpoints.
METRICS_S3="${METRICS_S3:-s3://edullm-checkpoints/token-sel/rho-1/metrics}"
TASK_LOSS_S3="${TASK_LOSS_S3:-s3://edullm-checkpoints/token-sel/rho-1/task_loss_results}"
OLMO_CORE_DIR="${OLMO_CORE_DIR:-$WORK/OLMo-core}"
OLMO_CORE_REVISION="${OLMO_CORE_REVISION:-99e0009ed67679c90da970ec5ba439c9459e3757}"
CFG_REL="${CFG_REL:-rho-1/configs/run_rho_10b.yaml}"
EXPORT_REF_PY="${EXPORT_REF_PY:-$TS_ROOT/reference/export_refhq_reference.py}"
ENQUEUE_TASK_LOSS="${ENQUEUE_TASK_LOSS:-$SCRIPT_DIR/enqueue_task_loss.sh}"
RDZV_PORT="${RDZV_PORT:-29521}"

OUT_DIR="$WORK/data/rho_10b"
TOK_DIR="$OUT_DIR/tokens"
CKPT_LOCAL="$OUT_DIR/checkpoints/rho_excess"
REF_DIR="$WORK/reference"
REF_PT="$REF_DIR/refhq_step1315_model.pt"
RUNTIME_CFG="$WORK/run_rho_10b.runtime.yaml"
LOG_DIR="$WORK/logs"
PROGRESS_DIR="$WORK/progress"

log() { echo "[$(date -Is)] $*"; }

verify_distcp_checkpoint() {
  local step_dir="$1"
  local meta="$step_dir/model_and_optim/.metadata"
  if [[ ! -f "$meta" ]]; then
    echo "WARN: incomplete checkpoint (no .metadata): $step_dir" >&2
    return 1
  fi
  local n
  n="$(find "$step_dir/model_and_optim" -maxdepth 1 -name '*.distcp' | wc -l | tr -d ' ')"
  if [[ "${n:-0}" -lt 1 ]]; then
    echo "WARN: checkpoint has no .distcp shards: $step_dir" >&2
    return 1
  fi
  return 0
}

# Sync permanent ladder under …/checkpoints[/method]/stepN → S3.
# Saves live at OUT_DIR/checkpoints/rho_excess/step*; preserve method subdir on S3.
sync_checkpoints_to_s3() {
  local src="$1" dst="$2"
  mkdir -p "$src"
  local step_dir rel
  shopt -s nullglob
  local candidates=("$src"/step* "$src"/*/step*)
  shopt -u nullglob
  for step_dir in "${candidates[@]}"; do
    [[ -d "$step_dir" ]] || continue
    if verify_distcp_checkpoint "$step_dir"; then
      rel="${step_dir#"$src"/}"
      aws s3 sync "$step_dir" "$dst/$rel" --only-show-errors || true
    else
      echo "SKIP S3 sync (incomplete): $step_dir" >&2
    fi
  done
  if [[ -f "$src/run_fingerprint.json" ]]; then
    aws s3 cp "$src/run_fingerprint.json" "$dst/run_fingerprint.json" --only-show-errors || true
  fi
  shopt -s nullglob
  local fps=("$src"/*/run_fingerprint.json)
  shopt -u nullglob
  for fp in "${fps[@]}"; do
    [[ -f "$fp" ]] || continue
    rel="${fp#"$src"/}"
    aws s3 cp "$fp" "$dst/$rel" --only-show-errors || true
  done
}

OFFLINE="${OFFLINE:-0}"
SKIP_S3_UPLOAD="${SKIP_S3_UPLOAD:-$OFFLINE}"

if [[ "$OFFLINE" == "1" ]]; then
  log "OFFLINE=1 — clearing AWS env; S3 sync/upload disabled"
  unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE AWS_SESSION_ENV || true
  export AWS_EC2_METADATA_DISABLED=true
fi

mkdir -p "$WORK" "$LOG_DIR" "$PROGRESS_DIR" "$REF_DIR" "$TOK_DIR" "$CKPT_LOCAL"

if [[ -x "$WORK/venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$WORK/venv/bin/activate"
elif [[ -x "$EDULLM_ROOT/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$EDULLM_ROOT/.venv/bin/activate"
fi

if [[ "$OFFLINE" != "1" && -f "${AWS_SESSION_ENV:-$WORK/aws-session.env}" ]]; then
  # shellcheck disable=SC1090
  source "${AWS_SESSION_ENV:-$WORK/aws-session.env}"
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OLMO_SHARED_FS=1
export TORCH_DIST_INIT_BARRIER=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OLMO_ATTN_BACKEND="$ATTN_BACKEND"
export WANDB_DISABLED=1
export WANDB_MODE=disabled
export TOKEN_SELECTION_TASK_LOSS_EVAL_SCRIPT="${TOKEN_SELECTION_TASK_LOSS_EVAL_SCRIPT:-$ENQUEUE_TASK_LOSS}"

python3 - <<PY
gbs, mbz, ng = 4_194_304, int("${RANK_MICROBATCH_SIZE}"), int("${NUM_GPUS}")
if ng * mbz == 0 or gbs % (ng * mbz) != 0:
    raise SystemExit(f"GBS {gbs} not divisible by NUM_GPUS*rank_mbz ({ng}*{mbz})")
print(f"ok: gpus={ng} rank_mbz={mbz} accum/rank={gbs // (ng * mbz)}")
PY

prepare() {
  CFG="$REPO_ROOT/$CFG_REL"
  if [[ ! -f "$CFG" ]]; then
    echo "ERROR: missing config $CFG" >&2
    exit 2
  fi
  if [[ ! -f "$EXPORT_REF_PY" ]]; then
    echo "ERROR: missing $EXPORT_REF_PY" >&2
    exit 2
  fi
  if [[ ! -d "$REPO_ROOT/token_selection" ]]; then
    echo "ERROR: missing $REPO_ROOT/token_selection (set EDULLM_ROOT / REPO_ROOT)" >&2
    exit 2
  fi

  if [[ ! -d "$OLMO_CORE_DIR/src/olmo_core" && ! -d "$OLMO_CORE_DIR/.git" ]]; then
    if [[ "$OFFLINE" == "1" ]]; then
      echo "ERROR: OFFLINE=1 but missing local OLMo-core at $OLMO_CORE_DIR" >&2
      exit 2
    fi
    git clone https://github.com/edu-llm/OLMo-core "$OLMO_CORE_DIR" \
      || git clone https://github.com/allenai/OLMo-core "$OLMO_CORE_DIR"
    git -C "$OLMO_CORE_DIR" fetch --all --tags || true
    git -C "$OLMO_CORE_DIR" checkout "$OLMO_CORE_REVISION" \
      || echo "WARN: pinned revision checkout failed; using HEAD"
  fi
  if ! python3 -c "from olmo_core.distributed.checkpoint import unshard_checkpoint" 2>/dev/null; then
    log "prepare: pip install editable OLMo-core + deps"
    python3 -m pip install -q -e "$OLMO_CORE_DIR" 'PyYAML>=6.0' || \
      python -m pip install -q -e "$OLMO_CORE_DIR" 'PyYAML>=6.0'
  else
    log "prepare: olmo_core already importable"
  fi

  log "prepare: RegMix tokens"
  if [[ -f "$TOK_DIR/paths.txt" ]] && find "$TOK_DIR" -name '*.npy' -print -quit | grep -q .; then
    log "SKIP train sync — reusing local $TOK_DIR"
  elif [[ "$OFFLINE" == "1" ]]; then
    echo "ERROR: OFFLINE=1 but missing preloaded tokens at $TOK_DIR" >&2
    exit 2
  else
    aws s3 sync "$TRAIN_DATA_S3" "$TOK_DIR" --only-show-errors
  fi

  log "prepare: RefHQ reference -> $REF_PT"
  if [[ -f "$REF_PT" ]]; then
    log "SKIP ref export — reusing $REF_PT"
  elif [[ "$OFFLINE" == "1" || "${SKIP_REF_EXPORT:-0}" == "1" ]]; then
    echo "ERROR: missing reference model $REF_PT (required offline)" >&2
    exit 2
  else
    python3 "$EXPORT_REF_PY" \
      --s3-uri "$REF_CKPT_S3" \
      --work-dir "$REF_DIR/work" \
      --output "$REF_PT" \
      2>&1 | tee "$LOG_DIR/export_refhq.log"
  fi
  test -f "$REF_PT"

  # Leave cuda_visible_devices empty so Slurm / outer CUDA_VISIBLE_DEVICES wins.
  python3 - "$CFG" "$RUNTIME_CFG" "$REF_PT" "$OUT_DIR" \
    "$NUM_GPUS" "$RANK_MICROBATCH_SIZE" "$NUM_WORKERS" "$DP_TYPE" "$ATTN_BACKEND" "$COMPILE_MODEL" <<'PY'
import sys
from pathlib import Path
import yaml
(
    src, dst, ref_pt, out_dir, num_gpus, rank_mbz, num_workers, dp_type, attn, compile_model
) = sys.argv[1:11]
cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
cfg.setdefault("reference", {})["load_path"] = ref_pt
cfg["output_dir"] = out_dir
cfg["arm"] = "rho-1"
train = cfg.setdefault("train", {})
train["cuda_visible_devices"] = ""
train["num_gpus"] = int(num_gpus)
train["rank_microbatch_size"] = int(rank_mbz)
train["num_workers"] = int(num_workers)
train["dp_type"] = dp_type
train["attn_backend"] = attn
train["compile_model"] = compile_model.strip() in {"1", "true", "True", "yes"}
# Enforce new-run contract in the runtime copy.
cfg["run_id"] = "rho-1-regmix10b-v1"
cfg["t0_steps"] = 0
cfg["t0_frac"] = 0
cfg["k"] = 0.6
train["checkpoint_every_steps"] = 125
train["checkpoint_keep_last"] = None
train["ephemeral_checkpoint_every_steps"] = None
train["save_async"] = False
train["max_grad_norm"] = 1.0
s3 = cfg.setdefault("s3", {})
s3["prefix"] = "token-sel/rho-1"
s3["checkpoint_bucket"] = "edullm-checkpoints"
eval_tl = cfg.setdefault("eval", {}).setdefault("task_loss", {})
eval_tl["enabled"] = True
eval_tl["results_dir"] = "task_loss_results/rho-1"
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"wrote {dst}")
print(f"  reference.load_path={ref_pt}")
print(f"  output_dir={out_dir}")
print(f"  num_gpus={num_gpus} rank_mbz={rank_mbz} dp={dp_type} compile={train['compile_model']}")
PY

  log "prepare: build_token_manifest + freeze_order"
  python3 -m token_selection.scripts.build_token_manifest --config "$RUNTIME_CFG" \
    2>&1 | tee "$LOG_DIR/build_token_manifest.log"
  python3 -m token_selection.scripts.freeze_order --config "$RUNTIME_CFG" \
    2>&1 | tee "$LOG_DIR/freeze_order.log"
  python3 -m token_selection.scripts.validate_experiment \
    --config "$RUNTIME_CFG" \
    --olmo-root "$OLMO_CORE_DIR" \
    2>&1 | tee "$LOG_DIR/validate_experiment.log"

  log "prepare complete → $RUNTIME_CFG ; ckpt_dir=$CKPT_LOCAL FROM_SCRATCH=$FROM_SCRATCH RESUME=$RESUME"
}

scratch_reset() {
  log "scratch_reset: clearing prior checkpoints in $CKPT_LOCAL (new run_id)"
  find "$CKPT_LOCAL" -mindepth 1 -maxdepth 1 -name 'step*' -exec rm -rf {} + 2>/dev/null || true
  rm -f "$CKPT_LOCAL/run_fingerprint.json"
}

train() {
  if [[ ! -f "$RUNTIME_CFG" ]]; then
    echo "ERROR: missing $RUNTIME_CFG — run prepare first" >&2
    exit 2
  fi

  # Refuse the discarded prior run_id if somehow present in the runtime YAML.
  python3 - "$RUNTIME_CFG" "$RUN_ID" "$DISCARDED_RUN_ID" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
rid = str(cfg.get("run_id") or "")
want, bad = sys.argv[2], sys.argv[3]
if rid == bad:
    raise SystemExit(f"refusing discarded run_id={bad!r}; use {want!r}")
if rid != want:
    raise SystemExit(f"run_id={rid!r} != expected {want!r}")
PY

  EXTRA_ARGS=()
  if [[ "$RESUME" == "1" ]]; then
    log "RESUME=1 — fingerprint-gated resume under run_id=$RUN_ID (not $DISCARDED_RUN_ID)"
    EXTRA_ARGS+=(--resume)
  elif [[ "$FROM_SCRATCH" == "1" ]]; then
    scratch_reset
  else
    echo "ERROR: set FROM_SCRATCH=1 (scratch rebuild) or RESUME=1 (crash resume of $RUN_ID)" >&2
    exit 2
  fi

  export TOKEN_SELECTION_SKIP_IDLE_CHECK="${TOKEN_SELECTION_SKIP_IDLE_CHECK:-1}"
  n_vis="$(python3 - <<'PY'
import os
print(len([p for p in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if p.strip()!=""]))
PY
)"
  if [[ "${n_vis}" != "0" && "${n_vis}" != "$NUM_GPUS" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES has $n_vis entries but NUM_GPUS=$NUM_GPUS" >&2
    exit 2
  fi

  log "train: torchrun nproc=$NUM_GPUS run_id=$RUN_ID FROM_SCRATCH=$FROM_SCRATCH RESUME=$RESUME"
  nvidia-smi || true
  date -Is > "$PROGRESS_DIR/heartbeat.txt"

  SYNC_PID=""
  if [[ "${SKIP_S3_UPLOAD}" != "1" ]]; then
    (
      while true; do
        sleep "${SYNC_INTERVAL_SEC:-120}"
        # Preserve method subdirs under token-sel/rho-1/checkpoints/
        sync_checkpoints_to_s3 "$OUT_DIR/checkpoints" "$CKPT_S3/checkpoints"
        aws s3 sync "$PROGRESS_DIR" "$CKPT_S3/progress" --only-show-errors || true
        if [[ -d "$OUT_DIR/metrics/rho_excess" ]]; then
          aws s3 sync "$OUT_DIR/metrics/rho_excess" "$METRICS_S3" --only-show-errors || true
        fi
        TL_LOCAL="${TASK_LOSS_LOCAL:-$TS_ROOT/task_loss_results/rho-1}"
        if [[ -d "$TL_LOCAL" ]]; then
          aws s3 sync "$TL_LOCAL" "$TASK_LOSS_S3" --only-show-errors || true
        fi
        date -Is > "$PROGRESS_DIR/heartbeat.txt" || true
      done
    ) &
    SYNC_PID=$!
    trap 'kill $SYNC_PID 2>/dev/null || true' EXIT
  else
    log "SKIP_S3_UPLOAD=1 — checkpoints stay local only"
    (
      while true; do
        sleep "${SYNC_INTERVAL_SEC:-120}"
        date -Is > "$PROGRESS_DIR/heartbeat.txt" || true
      done
    ) &
    SYNC_PID=$!
    trap 'kill $SYNC_PID 2>/dev/null || true' EXIT
  fi

  torchrun \
    --nnodes=1 \
    --nproc-per-node="$NUM_GPUS" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${RDZV_PORT}" \
    -m token_selection.scripts.train_olmo_template \
    --config "$RUNTIME_CFG" \
    --method rho_excess \
    --olmo-root "$OLMO_CORE_DIR" \
    --launch \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    2>&1 | tee "$LOG_DIR/train.log"

  if [[ "${SKIP_S3_UPLOAD}" != "1" ]]; then
    log "train exited; final S3 sync under $CKPT_S3"
    sync_checkpoints_to_s3 "$OUT_DIR/checkpoints" "$CKPT_S3/checkpoints"
    aws s3 sync "$PROGRESS_DIR" "$CKPT_S3/progress" --only-show-errors || true
    if [[ -d "$OUT_DIR/metrics/rho_excess" ]]; then
      aws s3 sync "$OUT_DIR/metrics/rho_excess" "$METRICS_S3" --only-show-errors || true
    fi
    TL_LOCAL="${TASK_LOSS_LOCAL:-$TS_ROOT/task_loss_results/rho-1}"
    if [[ -d "$TL_LOCAL" ]]; then
      aws s3 sync "$TL_LOCAL" "$TASK_LOSS_S3" --only-show-errors || true
    fi
  else
    log "train exited; SKIP final S3 sync"
  fi
}

case "$MODE" in
  prepare) prepare ;;
  train) train ;;
  all) prepare; train ;;
  *) echo "usage: $0 {prepare|train|all}" >&2; exit 2 ;;
esac
