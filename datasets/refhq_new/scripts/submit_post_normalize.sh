#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck disable=SC1091
source "${RUN_DIR}/env.sh"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
# shellcheck source=lib.sh
source "${REFHQ_NEW_SCRIPTS}/lib.sh"
refhq_new_export_pythonpath "${RUN_DIR}"
cd "${RUN_DIR}"

"${VENV}/bin/python" "${REFHQ_NEW_SCRIPTS}/build_english_tasks.py" --plan "${PLAN}"
ENGLISH_TASKS="${SCRATCH_ROOT}/manifests/english_tasks.txt"
N_ENG=$(grep -cve '^[[:space:]]*$' "${ENGLISH_TASKS}" || true)
if [[ "${N_ENG}" -lt 1 ]]; then
  echo "ERROR: no english tasks in ${ENGLISH_TASKS}" >&2
  exit 1
fi
echo "english_tasks=${N_ENG} file=${ENGLISH_TASKS}"

COMMON_EXPORT="ALL,RUN_DIR=${RUN_DIR},VENV=${VENV},PLAN=${PLAN},SOURCE_LIST=${SOURCE_LIST},SCRATCH_ROOT=${SCRATCH_ROOT},REFHQ_NEW_SCRIPTS=${REFHQ_NEW_SCRIPTS},STAGE_DIR=${STAGE_DIR},S3_BUCKET=${S3_BUCKET},S3_PREFIX=${S3_PREFIX},SEED=${SEED},TOKENIZE_TASKS=${TOKENIZE_TASKS},ENGLISH_TASKS=${ENGLISH_TASKS},AWS_SESSION_ENV=${AWS_SESSION_ENV:-${RUN_DIR}/aws-session.env}"

ENG_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --array=0-$((N_ENG - 1)) \
  --chdir="${RUN_DIR}" \
  --export="${COMMON_EXPORT}" \
  "${REFHQ_NEW_SCRIPTS}/dolma_english_filter.sbatch")
echo "english_job=${ENG_JOB}"

HOLD_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterok:${ENG_JOB} \
  --chdir="${RUN_DIR}" \
  --export="${COMMON_EXPORT}" \
  "${REFHQ_NEW_SCRIPTS}/holdout_docs.sbatch")
echo "holdout_job=${HOLD_JOB}"

POST_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterok:${HOLD_JOB} \
  --partition=normal --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:15:00 \
  --job-name=refhqn-post \
  --output=logs/refhqn_post_holdout_%j.out \
  --error=logs/refhqn_post_holdout_%j.err \
  --chdir="${RUN_DIR}" \
  --export="${COMMON_EXPORT},SKIP_PUBLISH=${SKIP_PUBLISH:-0}" \
  --wrap="bash ${RUN_DIR}/submit_post_holdout.sh")
echo "post_holdout_job=${POST_JOB}"
