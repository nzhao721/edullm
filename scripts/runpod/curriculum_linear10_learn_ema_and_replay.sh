#!/usr/bin/env bash
# Post-hoc EMA then clean W&B replay for linear10-learn.
set -Eeuo pipefail

REPO_DIR=/workspace/OLMo-core
RUN_DIR=/workspace/edullm-runs/curriculum/linear10-learn
LOG=/workspace/curriculum-linear10-learn-ema-replay.log
SOURCE_RUN_ID=83b3001e8e3fa7a56309d9b40de8e4e8
NEW_RUN_NAME="curriculum-linear10-learn-runpod-20260804-123617-linear10-learn-clean"

log() { printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "${LOG}"; }

log "=== PHASE 1: post-hoc EMA ==="
bash /workspace/curriculum_linear10_learn_post_train_ema.sh 2>&1 | tee -a "${LOG}"

log "=== PHASE 2: clean W&B replay ==="
source /workspace/wandb-session.env
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm"

python3 /workspace/replay_curriculum_wandb_clean.py \
  --source-run-id "${SOURCE_RUN_ID}" \
  --task-loss-dir "${RUN_DIR}/progress/task_loss_results" \
  --run-name "${NEW_RUN_NAME}" \
  --run-id-out "${RUN_DIR}/progress/clean_wandb_run_id.txt" \
  --prev-id-out "${RUN_DIR}/progress/dirty_wandb_run_id.txt" \
  --notes "Clean replay: train history from source (no eval backfill pollution); full eval ladder from local JSON including step 2000; EMA at step 2385." \
  --upload-artifacts \
  2>&1 | tee -a "${LOG}"

log "=== DONE ==="
cat "${RUN_DIR}/progress/clean_wandb_run_id.txt"
