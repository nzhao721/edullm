#!/usr/bin/env bash
# Launch RHO-1 (rho_excess) on 1..N GPUs (hardware-agnostic).
#
# Required env:
#   OLMO_ROOT              — pinned edu-llm/OLMo-core checkout (revision in the YAML)
#
# Optional env:
#   REFERENCE_LOAD_PATH    — local RefHQ step1315 .pt; if unset, --launch auto-materializes
#                            from YAML reference.s3_uri into TOKEN_SELECTION_REF_CACHE
#   NPROC                 — torchrun nproc_per_node (default: # of visible GPUs, else 1)
#   CUDA_VISIBLE_DEVICES  — physical GPU pin (required outside Slurm unless YAML sets it)
#   CONFIG                — override YAML path
#   RESUME=1              — resume from latest matching save-folder fingerprint
#   TOKEN_SELECTION_SKIP_IDLE_CHECK=1 — skip idle GPU probe (Slurm sets this path too)
#   TOKEN_SELECTION_TASK_LOSS_EVAL_SCRIPT — enqueue script for post-save task_loss
#   TOKEN_SELECTION_REF_CACHE — shared DistCP→.pt cache (default: token-selection/.cache/refhq)
#
# Examples (from experiments/token-selection/):
#   CUDA_VISIBLE_DEVICES=0 NPROC=1 OLMO_ROOT=/path/OLMo-core bash rho-1/launch.sh
#   CUDA_VISIBLE_DEVICES=0 NPROC=1 \
#     REFERENCE_LOAD_PATH=/path/refhq_step1315_model.pt \
#     OLMO_ROOT=/path/OLMo-core bash rho-1/launch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM_ROOT="${SCRIPT_DIR}"
TS_ROOT="$(cd "${ARM_ROOT}/.." && pwd)"
export PYTHONPATH="${TS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_SRC="${CONFIG:-${ARM_ROOT}/configs/run_rho_10b.yaml}"
METHOD="${METHOD:-rho_excess}"
RUNTIME_CFG="${RUNTIME_CFG:-}"
REFERENCE_LOAD_PATH="${REFERENCE_LOAD_PATH:-}"

if [[ -z "${OLMO_ROOT:-}" ]]; then
  echo "OLMO_ROOT must point at the pinned OLMo-core checkout" >&2
  exit 1
fi
if [[ -n "${REFERENCE_LOAD_PATH}" && ! -e "${REFERENCE_LOAD_PATH}" ]]; then
  echo "REFERENCE_LOAD_PATH does not exist: ${REFERENCE_LOAD_PATH}" >&2
  exit 1
fi

if [[ -z "${NPROC:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _gpus <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC="${#_gpus[@]}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    NPROC="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    NPROC="${NPROC:-1}"
  else
    NPROC=1
  fi
fi
if [[ "${NPROC}" -lt 1 ]]; then
  echo "NPROC must be >= 1 (got ${NPROC})" >&2
  exit 1
fi

# Write a runtime YAML (never mutate the checked-in config). Optional REF inject;
# null load_path + s3_uri → materialize inside train_olmo_template --launch.
if [[ -z "${RUNTIME_CFG}" ]]; then
  RUNTIME_CFG="$(mktemp "${TMPDIR:-/tmp}/run_rho_10b.XXXXXX.yaml")"
fi
REFERENCE_LOAD_PATH="${REFERENCE_LOAD_PATH}" python3 - "${CONFIG_SRC}" "${RUNTIME_CFG}" <<'PY'
import os
import sys
from pathlib import Path
import yaml
src, dst = sys.argv[1:3]
cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
ref_pt = os.environ.get("REFERENCE_LOAD_PATH", "").strip()
if ref_pt:
    cfg.setdefault("reference", {})["load_path"] = ref_pt
cfg["run_id"] = "rho-1-regmix10b-v1"
cfg["arm"] = "rho-1"
cfg["t0_steps"] = 0
cfg["t0_frac"] = 0
cfg["k"] = 0.6
cfg.setdefault("s3", {})["prefix"] = "token-sel/rho-1"
train = cfg.setdefault("train", {})
train["checkpoint_every_steps"] = 125
train["checkpoint_keep_last"] = None
train["ephemeral_checkpoint_every_steps"] = None
train.setdefault("compile_model", True)
train.setdefault("max_grad_norm", 1.0)
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
lp = (cfg.get("reference") or {}).get("load_path")
print(f"runtime config -> {dst} (reference.load_path={lp!r}; auto-materialize if null)")
PY

export TOKEN_SELECTION_TASK_LOSS_EVAL_SCRIPT="${TOKEN_SELECTION_TASK_LOSS_EVAL_SCRIPT:-${ARM_ROOT}/farmshare/enqueue_task_loss.sh}"

LAUNCH_ARGS=(--config "${RUNTIME_CFG}" --method "${METHOD}" --olmo-root "${OLMO_ROOT}" --launch)
if [[ "${RESUME:-0}" == "1" ]]; then
  LAUNCH_ARGS+=(--resume)
fi

echo "rho-1 launch: NPROC=${NPROC} METHOD=${METHOD} RESUME=${RESUME:-0}"
exec python -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
  -m token_selection.scripts.train_olmo_template \
  "${LAUNCH_ARGS[@]}"
