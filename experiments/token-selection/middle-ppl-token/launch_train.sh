#!/usr/bin/env bash
# Middle-PPL token arm — hardware-agnostic train launcher (1…N GPUs).
#
# Env (all optional except OLMO_CORE_DIR / OLMO_ROOT for --launch):
#   EDULLM_ROOT          repo root (default: inferred from this script)
#   OLMO_CORE_DIR        pinned OLMo-core checkout (alias: OLMO_ROOT)
#   NUM_GPUS             torchrun nproc (default: CVD count, else nvidia-smi, else 1)
#   CUDA_VISIBLE_DEVICES physical GPU pin (required on bare multi-GPU hosts)
#   RANK_MICROBATCH_SIZE tokens/rank microbatch (default: YAML 65536)
#   MODE                 prepare | train | all  (default: all)
#   RESUME               1 to --resume
#   TASK_LOSS_EVAL       0 to disable post-save task_loss spawn
#   WORK                 local working dir for data/checkpoints (default: arm data dir)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EDULLM_ROOT="${EDULLM_ROOT:-$(cd "$TS_ROOT/../.." && pwd)}"
REPO_PY="${REPO_PY:-$TS_ROOT}"
CFG_SRC="${CFG_SRC:-$TS_ROOT/middle-ppl-token/configs/run_middle_ppl_token_10b.yaml}"
OLMO_CORE_DIR="${OLMO_CORE_DIR:-${OLMO_ROOT:-}}"
MODE="${MODE:-all}"
RESUME="${RESUME:-0}"
WORK="${WORK:-$TS_ROOT/middle-ppl-token/data/middle_ppl_token_10b}"

# Discover world size: explicit NUM_GPUS > CVD length > nvidia-smi > 1.
if [[ -n "${NUM_GPUS:-}" ]]; then
  NPROC="$NUM_GPUS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _gpus <<< "${CUDA_VISIBLE_DEVICES// /}"
  NPROC="${#_gpus[@]}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  NPROC="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  NPROC="${NPROC:-1}"
else
  NPROC=1
fi
if [[ "$NPROC" -lt 1 ]]; then
  echo "NUM_GPUS / CUDA_VISIBLE_DEVICES must yield >= 1 GPU" >&2
  exit 1
fi

export PYTHONPATH="${REPO_PY}${PYTHONPATH:+:$PYTHONPATH}"
export TOKEN_SELECTION_SKIP_IDLE_CHECK="${TOKEN_SELECTION_SKIP_IDLE_CHECK:-${SLURM_JOB_ID:+1}}"

log() { echo "[$(date -Is)] $*"; }

cd "$TS_ROOT"

# Always materialize a runtime YAML so WORK + optional microbatch overrides
# leave the fingerprinted recipe in git untouched.
mkdir -p "$WORK"
RUNTIME_CFG="${WORK}/run_middle_ppl_token_10b.runtime.yaml"
python - <<PY
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(r"$CFG_SRC").read_text(encoding="utf-8"))
train = cfg.setdefault("train", {})
cfg["output_dir"] = r"$WORK"
train["cuda_visible_devices"] = ""
if r"${RANK_MICROBATCH_SIZE:-}" != "":
    train["rank_microbatch_size"] = int("${RANK_MICROBATCH_SIZE}")
Path(r"$RUNTIME_CFG").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(r"$RUNTIME_CFG")
PY
CFG="$RUNTIME_CFG"

prepare() {
  log "sync tokens"
  python -m token_selection.scripts.sync_artifacts --config "$CFG" --direction download --what tokens
  log "build token manifest"
  python -m token_selection.scripts.build_token_manifest --config "$CFG"
  log "freeze order"
  python -m token_selection.scripts.freeze_order --config "$CFG"
  if [[ -n "$OLMO_CORE_DIR" ]]; then
    log "validate"
    python -m token_selection.scripts.validate_experiment --config "$CFG" --olmo-root "$OLMO_CORE_DIR"
  else
    log "skip validate (set OLMO_CORE_DIR to enable)"
  fi
}

train() {
  if [[ -z "$OLMO_CORE_DIR" ]]; then
    echo "OLMO_CORE_DIR (or OLMO_ROOT) required for train/--launch" >&2
    exit 1
  fi
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Set CUDA_VISIBLE_DEVICES to idle GPU index/list (or run under Slurm)." >&2
    exit 1
  fi
  # Fail fast if GBS is not divisible by world_size * rank_microbatch.
  python - <<'PY' "$CFG" "$NPROC"
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
gbs = int(cfg["train"]["global_batch_size"])
rmbs = int(cfg["train"]["rank_microbatch_size"])
ws = int(sys.argv[2])
if gbs % (ws * rmbs) != 0:
    raise SystemExit(
        f"global_batch_size={gbs} not divisible by world_size*rank_microbatch "
        f"({ws}*{rmbs}={ws * rmbs}). Set RANK_MICROBATCH_SIZE so it divides."
    )
print(f"batch ok: gbs={gbs} ws={ws} rmbs={rmbs} accum={gbs // (ws * rmbs)}")
PY
  local resume_flag=()
  if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
    resume_flag=(--resume)
  fi
  log "torchrun nproc=$NPROC cfg=$CFG"
  python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    -m token_selection.scripts.train_olmo_template \
    --config "$CFG" --method middle_ppl \
    --olmo-root "$OLMO_CORE_DIR" --launch "${resume_flag[@]}"
}

case "$MODE" in
  prepare) prepare ;;
  train) train ;;
  all) prepare; train ;;
  *) echo "MODE must be prepare|train|all (got $MODE)" >&2; exit 1 ;;
esac
