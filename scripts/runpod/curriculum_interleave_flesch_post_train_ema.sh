#!/usr/bin/env bash
# Post-training EMA merge + task-loss eval for curriculum interleave-flesch.
# Logs to the same W&B run as training (step 2385 eval point / final model).
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/workspace/OLMo-core}"
RUN_DIR="${RUN_DIR:-/workspace/edullm-runs/curriculum/interleave-flesch}"
ARM="${ARM:-interleave-flesch}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/workspace/wandb-session.env}"
EMA_LOG="${EMA_LOG:-/workspace/curriculum-interleave-flesch-ema.log}"
DONE_MARKER="${RUN_DIR}/progress/ema_integrated.done"
LEGACY_DONE_MARKER="${RUN_DIR}/progress/ema_post_train.done"
LOCK_FILE="${RUN_DIR}/progress/ema_post_train.lock"
FINAL_TASK_LOSS="${RUN_DIR}/progress/task_loss_results/step2384_task_loss.json"
EMA_TASK_LOSS="${RUN_DIR}/progress/task_loss_results/step2385_task_loss.json"
EMA_LEGACY_TASK_LOSS="${RUN_DIR}/checkpoints/step2384-ema/step2384-ema_task_loss.json"

log() { printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "${EMA_LOG}"; }

if [[ (-f "${DONE_MARKER}" || -f "${LEGACY_DONE_MARKER}") && (-f "${EMA_TASK_LOSS}" || -f "${EMA_LEGACY_TASK_LOSS}") ]]; then
  log "EMA post-train already complete"
  exit 0
fi
if [[ ! -f "${FINAL_TASK_LOSS}" ]]; then
  log "step-2384 task_loss missing; training not ready for EMA"
  exit 3
fi
if [[ -f "${LOCK_FILE}" ]]; then
  lock_age="$(( $(date +%s) - $(stat -c %Y "${LOCK_FILE}") ))"
  if (( lock_age < 7200 )); then
    log "EMA post-train already running (lock age ${lock_age}s)"
    exit 2
  fi
  log "stale EMA lock (${lock_age}s); reclaiming"
  rm -f "${LOCK_FILE}"
fi

umask 077
touch "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT

[[ -f "${WANDB_ENV_FILE}" ]] || { log "missing W&B env: ${WANDB_ENV_FILE}"; exit 4; }
# shellcheck disable=SC1090
source "${WANDB_ENV_FILE}"
[[ -f "${RUN_DIR}/run.env" ]] || { log "missing run identity: ${RUN_DIR}/run.env"; exit 4; }
# shellcheck disable=SC1090
source "${RUN_DIR}/run.env"

[[ -n "${WANDB_API_KEY:-}" ]] || { log "WANDB_API_KEY is required"; exit 4; }
[[ -n "${WANDB_RUN_ID:-}" ]] || { log "WANDB_RUN_ID is required"; exit 4; }

export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm"
export EDULLM_WANDB_PROJECT="${EDULLM_WANDB_PROJECT:-curriculum}"
export WANDB_PROJECT="${EDULLM_WANDB_PROJECT}"
export WANDB_RESUME=must

log "starting post-hoc EMA for ${ARM} (wandb run ${WANDB_RUN_ID})"
python3 "${REPO_DIR}/.edullm/curriculum_ema.py" \
  --checkpoints-root "${RUN_DIR}/checkpoints" \
  --arm "${ARM}" \
  --task-loss-eval-script "${REPO_DIR}/.edullm/task_loss/eval_task_loss_olmo_core.py" \
  --task-loss-nproc 8 \
  --wandb-mode online \
  --wandb-run-id "${WANDB_RUN_ID}" \
  --run-name "${EDULLM_RUN_ID:-}" \
  --wandb-entity "${WANDB_ENTITY:-}" \
  2>&1 | tee -a "${EMA_LOG}"

[[ -f "${EMA_TASK_LOSS}" || -f "${EMA_LEGACY_TASK_LOSS}" ]] || {
  log "EMA task_loss output missing: ${EMA_TASK_LOSS}"
  exit 5
}
mkdir -p "${RUN_DIR}/progress"
printf '%s\n' "{\"wandb_step\":2385,\"final_model\":\"ema\"}" > "${DONE_MARKER}"
touch "${LEGACY_DONE_MARKER}"
log "EMA post-train complete"
