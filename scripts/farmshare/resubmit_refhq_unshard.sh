#!/usr/bin/env bash
# Refresh AWS session (if possible) and resubmit step1315 unshard on FarmShare.
set -Eeuo pipefail

RUN_DIR="${RUN_DIR:-/scratch/users/nzhao2/agent-runs/refhq-unshard-20260727}"
EDULLM_ROOT="${EDULLM_ROOT:-/scratch/users/nzhao2/agent-runs/rho-excess-10b-l40s/edullm}"
FALLBACK_VENV="${FALLBACK_VENV:-/scratch/users/nzhao2/agent-runs/rho-excess-10b-l40s/venv}"

export PATH="$HOME/.local/bin:$HOME/tools/aws/bin:$PATH"
if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.nvm/nvm.sh"
  nvm use default >/dev/null 2>&1 || true
fi

PY=python3
for cand in "$RUN_DIR/venv/bin/python" "$FALLBACK_VENV/bin/python"; do
  if [[ -x "$cand" ]]; then
    PY="$cand"
    break
  fi
done

WRITER="$EDULLM_ROOT/scripts/farmshare/write_aws_session_env.py"
if [[ ! -f "$WRITER" ]]; then
  echo "ERROR: missing $WRITER" >&2
  exit 2
fi

export AWS_PROFILE=sbsandbox AWS_DEFAULT_REGION=us-east-1
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SESSION_ENV || true
"$PY" "$WRITER" --output "$RUN_DIR/aws-session.env" --profile sbsandbox --region us-east-1
chmod 600 "$RUN_DIR/aws-session.env"

set +u
# shellcheck disable=SC1090
source "$RUN_DIR/aws-session.env"
set -u
unset AWS_PROFILE
aws sts get-caller-identity --output text

cd "$RUN_DIR"
export RUN_DIR EDULLM_ROOT AWS_SESSION_ENV="$RUN_DIR/aws-session.env"
JOB=$(sbatch --exclude=wheat-01 \
  --export=ALL,RUN_DIR,EDULLM_ROOT,AWS_SESSION_ENV \
  "$RUN_DIR/unshard_refhq_step1315.sbatch" | awk '{print $NF}')
echo "SUBMITTED_JOB_ID=$JOB"
squeue -j "$JOB" || sacct -j "$JOB" --format=JobID,State,ExitCode -n | head -2
