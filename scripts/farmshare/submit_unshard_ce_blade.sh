#!/usr/bin/env bash
# Mint AWS session (if needed) and submit CE + BLADE unshard jobs on FarmShare.
set -Eeuo pipefail

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/users/nzhao2/agent-runs}"
REPO_SCRIPTS="${REPO_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

CE_RUN_DIR="${CE_RUN_DIR:-$SCRATCH_BASE/unshard-ce-regmix10b-$STAMP}"
BLADE_RUN_DIR="${BLADE_RUN_DIR:-$SCRATCH_BASE/unshard-blade-regmix10b-$STAMP}"

CE_S3="${CE_S3:-s3://edullm-checkpoints/olmo-370m/edullm-370M-ce-regmix10b/checkpoints/}"
BLADE_S3="${BLADE_S3:-s3://edullm-checkpoints/olmo-370m/edullm-370M-blade-regmix10b/checkpoints/}"
STEPS="${STEPS:-250 500 750 1000 1250 1500 1750 2000 2250 2384}"

for d in "$CE_RUN_DIR" "$BLADE_RUN_DIR"; do
  mkdir -p "$d/logs"
  cp -f "$REPO_SCRIPTS/unshard_olmo_core_run.sh" "$d/"
  cp -f "$REPO_SCRIPTS/unshard_distcp_to_model_pt.py" "$d/"
  cp -f "$REPO_SCRIPTS/unshard_ce_blade.sbatch" "$d/"
  chmod +x "$d/unshard_olmo_core_run.sh"
done

# Reuse shared AWS session writer if present.
RESUBMIT="${RESUBMIT:-$SCRATCH_BASE/refhq-unshard-20260727/resubmit_refhq_unshard.sh}"
if [[ -f "$RESUBMIT" ]]; then
  RUN_DIR="$CE_RUN_DIR" bash -c '
    set -euo pipefail
    export PATH="$HOME/.local/bin:$HOME/tools/aws/bin:$PATH"
    if [[ -s "$HOME/.nvm/nvm.sh" ]]; then source "$HOME/.nvm/nvm.sh"; fi
    PY=/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/venv/bin/python
    WRITER=/scratch/users/nzhao2/agent-runs/rho-excess-10b-l40s/edullm/scripts/farmshare/write_aws_session_env.py
    export AWS_PROFILE=sbsandbox AWS_DEFAULT_REGION=us-east-1
    "$PY" "$WRITER" --output "$RUN_DIR/aws-session.env" --profile sbsandbox --region us-east-1
    chmod 600 "$RUN_DIR/aws-session.env"
    cp -f "$RUN_DIR/aws-session.env" "'"$BLADE_RUN_DIR"'/aws-session.env"
    chmod 600 "'"$BLADE_RUN_DIR"'/aws-session.env"
  '
else
  for d in "$CE_RUN_DIR" "$BLADE_RUN_DIR"; do
    if [[ -f /scratch/users/nzhao2/agent-runs/aws-session.env ]]; then
      cp -f /scratch/users/nzhao2/agent-runs/aws-session.env "$d/aws-session.env"
    fi
  done
fi

submit_one() {
  local run_dir=$1 label=$2 s3=$3
  cd "$run_dir"
  export RUN_DIR="$run_dir" RUN_LABEL="$label" S3_CKPT_PREFIX="$s3" STEPS AWS_SESSION_ENV="$run_dir/aws-session.env"
  local job
  job=$(sbatch --exclude=wheat-01 \
    --job-name="unshard-${label}" \
    --export=ALL,RUN_DIR,RUN_LABEL,S3_CKPT_PREFIX,STEPS,AWS_SESSION_ENV \
    "$run_dir/unshard_ce_blade.sbatch" | awk '{print $NF}')
  echo "$job" > "$run_dir/job_id.txt"
  echo "SUBMITTED $label job=$job dir=$run_dir"
}

submit_one "$CE_RUN_DIR" "ce-regmix10b" "$CE_S3"
submit_one "$BLADE_RUN_DIR" "blade-regmix10b" "$BLADE_S3"
squeue --me
