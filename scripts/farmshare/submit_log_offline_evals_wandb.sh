#!/usr/bin/env bash
# CPU-only: backfill offline eval JSONs into the original W&B run (one session).
set -Eeuo pipefail

RUN_DIR="${RUN_DIR:-/scratch/users/nzhao2/agent-runs/smollm2-135m-750m-27ep-fresh}"
EVAL_DIR="${EVAL_DIR:-${RUN_DIR}/output/evals_piqa_obqa}"
STAGING="${STAGING:-/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging}"
VENV="${VENV:-/scratch/users/nzhao2/agent-runs/venvs/smollm2-train}"
LOG_PY="${LOG_PY:-${STAGING}/scripts/farmshare/log_offline_evals_to_wandb.py}"
WANDB_PROJECT="${WANDB_PROJECT:-edullm-smollm2}"
WANDB_ENTITY="${WANDB_ENTITY:-eduLLM}"
WANDB_RUN_ID="${WANDB_RUN_ID:-$(tr -d ' \t\r\n' < "${RUN_DIR}/output/wandb_run_id.txt")}"
DEPENDENCY="${DEPENDENCY:-}"
JOB_NAME="${JOB_NAME:-sm2-wb-backfill}"

mkdir -p "${RUN_DIR}/logs"

if [[ ! -f "${RUN_DIR}/wandb-session.env" ]]; then
  echo "missing ${RUN_DIR}/wandb-session.env" >&2
  exit 2
fi
if [[ ! -f "${LOG_PY}" ]]; then
  echo "missing ${LOG_PY}" >&2
  exit 2
fi

SBATCH="${RUN_DIR}/logs/wandb_backfill.sbatch"
cat > "${SBATCH}" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --exclude=wheat-01
#SBATCH --chdir=${RUN_DIR}
#SBATCH --output=${RUN_DIR}/logs/wandb-backfill-%j.out
#SBATCH --error=${RUN_DIR}/logs/wandb-backfill-%j.err
$(if [[ -n "${DEPENDENCY}" ]]; then echo "#SBATCH --dependency=${DEPENDENCY}"; fi)

set -Eeuo pipefail
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load python/3.12.3 2>/dev/null || true
source "${VENV}/bin/activate"
# shellcheck disable=SC1091
source "${RUN_DIR}/wandb-session.env"
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1

python "${LOG_PY}" \\
  --eval-dir "${EVAL_DIR}" \\
  --wandb-project "${WANDB_PROJECT}" \\
  --wandb-entity "${WANDB_ENTITY}" \\
  --wandb-run-id "${WANDB_RUN_ID}" \\
  --upload-artifacts
EOF

JOB_ID=$(sbatch --parsable "${SBATCH}")
echo "job_id=${JOB_ID}"
echo "WANDB_RUN_ID=${WANDB_RUN_ID}"
echo "EVAL_DIR=${EVAL_DIR}"
echo "DEPENDENCY=${DEPENDENCY:-none}"
echo "${JOB_ID}" > "${RUN_DIR}/logs/wandb_backfill_job_id.txt"
