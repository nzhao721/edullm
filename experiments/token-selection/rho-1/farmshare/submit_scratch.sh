#!/usr/bin/env bash
# Submit a scratch RHO-1 job (rho-1-regmix10b-v1). Never resumes discarded v1.
#
# Ephemeral empty-scratch: stage edullm-data; durable ckpts → W&B artifacts.
# Do not assume a pre-populated RUN_DIR (tokens/venv/ckpts) survives wipe.
#
#   export RUN_DIR=/scratch/users/$USER/agent-runs/rho-1-regmix10b-v1
#   export EDULLM_ROOT=/path/to/edullm
#   export NUM_GPUS=4
#   # Optional: push aws-session.env into RUN_DIR for S3 stage/export
#   bash $EDULLM_ROOT/experiments/token-selection/rho-1/farmshare/submit_scratch.sh
#
# Crash resume of THIS run_id only (fetches from S3 if local save_folder empty):
#   RESUME=1 bash .../submit_scratch.sh
#
# Local smoke only: OFFLINE=1 WANDB_MODE=disabled
set -euo pipefail

RUN_DIR="${RUN_DIR:?set RUN_DIR}"
EDULLM_ROOT="${EDULLM_ROOT:?set EDULLM_ROOT}"
SBATCH="${EDULLM_ROOT}/experiments/token-selection/rho-1/farmshare/train_rho.sbatch"
NUM_GPUS="${NUM_GPUS:-4}"

export RUN_DIR EDULLM_ROOT NUM_GPUS
export RANK_MICROBATCH_SIZE="${RANK_MICROBATCH_SIZE:-16384}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export COMPILE_MODEL="${COMPILE_MODEL:-1}"
export RESUME="${RESUME:-0}"
if [[ "${RESUME}" == "1" ]]; then
  export FROM_SCRATCH=0
else
  export FROM_SCRATCH=1
fi
export TOKEN_SELECTION_SKIP_IDLE_CHECK=1
export SKIP_REF_EXPORT="${SKIP_REF_EXPORT:-0}"
export OFFLINE="${OFFLINE:-0}"

cd "$RUN_DIR"
mkdir -p logs

# Clear leftover checkpoints only for scratch starts (not crash resume).
CKPT_LOCAL="$RUN_DIR/data/rho_10b/checkpoints/rho_excess"
if [[ "${FROM_SCRATCH}" == "1" && -d "$CKPT_LOCAL" ]]; then
  echo "FROM_SCRATCH=1: clearing $CKPT_LOCAL/step*"
  find "$CKPT_LOCAL" -mindepth 1 -maxdepth 1 -name 'step*' -exec rm -rf {} + 2>/dev/null || true
  rm -f "$CKPT_LOCAL/run_fingerprint.json"
fi

JOB=$(sbatch --exclude=wheat-01 \
  --gres=gpu:"${NUM_GPUS}" \
  --export=ALL,RUN_DIR,EDULLM_ROOT,NUM_GPUS,RANK_MICROBATCH_SIZE,NUM_WORKERS,COMPILE_MODEL,FROM_SCRATCH,RESUME,TOKEN_SELECTION_SKIP_IDLE_CHECK,SKIP_REF_EXPORT,OFFLINE \
  "$SBATCH" | awk '{print $NF}')
echo "$JOB" > "$RUN_DIR/job_id.txt"
echo "SUBMITTED_JOB_ID=$JOB run_id=rho-1-regmix10b-v1 FROM_SCRATCH=${FROM_SCRATCH} RESUME=${RESUME} OFFLINE=${OFFLINE}"
squeue -u "$(whoami)" -o '%.18i %.16j %.8T %.10M %R' | head -10
