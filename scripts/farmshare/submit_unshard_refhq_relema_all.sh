#!/usr/bin/env bash
# Submit full RefHQ + REL-EMA distcp → model.pt unshard jobs on FarmShare.
set -Eeuo pipefail

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/users/nzhao2/agent-runs}"
REPO_SCRIPTS="${REPO_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

REFHQ_RUN_DIR="${REFHQ_RUN_DIR:-$SCRATCH_BASE/refhq-models-all-$STAMP}"
RELEMA_RUN_DIR="${RELEMA_RUN_DIR:-$SCRATCH_BASE/relema-models-all-$STAMP}"

REFHQ_S3="${REFHQ_S3:-s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/}"
RELEMA_S3="${RELEMA_S3:-s3://edullm-checkpoints/token-selection/rel-ema-10b-scratch-v1/rel_ema/}"
RELEMA_LOCAL="${RELEMA_LOCAL:-/scratch/users/nzhao2/checkpoints/token-selection-370m/rel-ema}"

REFHQ_STEPS="${REFHQ_STEPS:-125 250 375 500 625 750 875 1000 1125 1315}"
RELEMA_STEPS="${RELEMA_STEPS:-0 24 125 250 500 750 1000 1250 1500 1750 2000 2250 2375 2386}"

for d in "$REFHQ_RUN_DIR" "$RELEMA_RUN_DIR"; do
  mkdir -p "$d/logs" "$d/distcp" "$d/unsharded"
  cp -f "$REPO_SCRIPTS/unshard_olmo_core_run.sh" "$d/"
  cp -f "$REPO_SCRIPTS/unshard_distcp_to_model_pt.py" "$d/"
  cp -f "$REPO_SCRIPTS/unshard_all_distcp.sbatch" "$d/"
  chmod +x "$d/unshard_olmo_core_run.sh"
done

# Seed already-unsharded finals so jobs skip them.
if [[ -f /scratch/users/nzhao2/agent-runs/refhq-unshard-20260727/reference/refhq_step1315_model.pt ]]; then
  mkdir -p "$REFHQ_RUN_DIR/unsharded/step1315"
  cp -n /scratch/users/nzhao2/agent-runs/refhq-unshard-20260727/reference/refhq_step1315_model.pt \
    "$REFHQ_RUN_DIR/unsharded/step1315/model.pt" || true
fi
if [[ -f /scratch/users/nzhao2/agent-runs/relema-unshard-20260727/relema_step2386_model.pt ]]; then
  mkdir -p "$RELEMA_RUN_DIR/unsharded/step2386"
  cp -n /scratch/users/nzhao2/agent-runs/relema-unshard-20260727/relema_step2386_model.pt \
    "$RELEMA_RUN_DIR/unsharded/step2386/model.pt" || true
fi

# Mint AWS session for RefHQ S3 sync.
PY=/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/venv/bin/python
WRITER=/scratch/users/nzhao2/agent-runs/rho-excess-10b-l40s/edullm/scripts/farmshare/write_aws_session_env.py
export PATH="$HOME/.local/bin:$HOME/tools/aws/bin:$PATH"
if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.nvm/nvm.sh"
fi
export AWS_PROFILE=sbsandbox AWS_DEFAULT_REGION=us-east-1
"$PY" "$WRITER" --output "$REFHQ_RUN_DIR/aws-session.env" --profile sbsandbox --region us-east-1
chmod 600 "$REFHQ_RUN_DIR/aws-session.env"

submit_one() {
  local run_dir=$1 label=$2 export_extra=$3
  cd "$run_dir"
  export RUN_DIR="$run_dir" RUN_LABEL="$label"
  # shellcheck disable=SC2086
  local job
  job=$(sbatch --exclude=wheat-01 \
    --job-name="unshard-${label}" \
    --export=ALL,RUN_DIR,RUN_LABEL${export_extra} \
    "$run_dir/unshard_all_distcp.sbatch" | awk '{print $NF}')
  echo "$job" > "$run_dir/job_id.txt"
  echo "SUBMITTED $label job=$job dir=$run_dir"
}

# RefHQ: sync from S3 then unshard.
export S3_CKPT_PREFIX="$REFHQ_S3"
export STEPS="$REFHQ_STEPS"
export AWS_SESSION_ENV="$REFHQ_RUN_DIR/aws-session.env"
export SKIP_S3_SYNC=0
export DISTCP_ROOT="$REFHQ_RUN_DIR/distcp"
export OUT_ROOT="$REFHQ_RUN_DIR/unsharded"
submit_one "$REFHQ_RUN_DIR" "refhq-all" ",S3_CKPT_PREFIX,STEPS,AWS_SESSION_ENV,SKIP_S3_SYNC,DISTCP_ROOT,OUT_ROOT"

# REL-EMA: already local on FarmShare; skip S3.
export S3_CKPT_PREFIX="$RELEMA_S3"
export STEPS="$RELEMA_STEPS"
export SKIP_S3_SYNC=1
export DISTCP_ROOT="$RELEMA_LOCAL"
export OUT_ROOT="$RELEMA_RUN_DIR/unsharded"
unset AWS_SESSION_ENV || true
submit_one "$RELEMA_RUN_DIR" "relema-all" ",S3_CKPT_PREFIX,STEPS,SKIP_S3_SYNC,DISTCP_ROOT,OUT_ROOT"

squeue -u "$(whoami)"
