#!/usr/bin/env bash
# Upload any on-disk MixLaw checkpoints (and progress JSON) to the existing W&B run.
set -Eeuo pipefail

RUN_DIR="${RUN_DIR:-/workspace/mixlaw-370m-mix01}"
REPO_DIR="${REPO_DIR:-/workspace/edullm}"
WANDB_ENV="${WANDB_ENV:-/workspace/wandb-session.env}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-4194304}"

if [[ -f "${WANDB_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${WANDB_ENV}"
fi

export PYTHONPATH="${REPO_DIR}/experiments/skill-dag:${REPO_DIR}/experiments/skill-dag/mixlaw:${REPO_DIR}/experiments/token-selection:${PYTHONPATH:-}"
export RUN_DIR REPO_DIR TOKENS_PER_STEP

python3 <<'PY'
import os
import sys
from argparse import Namespace
from pathlib import Path

import wandb

RUN_DIR = Path(os.environ["RUN_DIR"])
REPO_DIR = Path(os.environ["REPO_DIR"])
progress = RUN_DIR / "progress"
ckpt_root = RUN_DIR / "save" / "checkpoints"
task_loss = RUN_DIR / "task_loss_results"
tokens_per_step = int(os.environ.get("TOKENS_PER_STEP", "4194304"))

sys.path.insert(0, str(REPO_DIR / "experiments" / "skill-dag" / "mixlaw"))
from mixlaw_wandb import finish_wandb, wandb_upload_existing  # noqa: E402

id_path = progress / "wandb_run_id.txt"
if not id_path.is_file():
    raise SystemExit(f"missing wandb run id: {id_path}")

run_id = id_path.read_text(encoding="utf-8").strip()
ckpts = sorted(p for p in ckpt_root.glob("step*") if p.is_dir())
print(f"[upload] resuming wandb run={run_id} checkpoints_on_disk={len(ckpts)}")
for p in ckpts:
    print(f"[upload] local {p.name}")

run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "mixlaw"),
    entity=os.environ.get("WANDB_ENTITY") or None,
    group=os.environ.get("WANDB_GROUP") or None,
    id=run_id,
    resume="must",
    dir=str(progress / "wandb"),
)
wandb_upload_existing(
    run,
    checkpoints_root=ckpt_root,
    task_loss_dir=task_loss if task_loss.is_dir() else None,
    progress_dir=progress,
    tokens_per_step=tokens_per_step,
)
finish_wandb(run)
print("UPLOAD_DONE")
PY
