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

TASKS="${TOKENIZE_TASKS:-${SCRATCH_ROOT}/manifests/tokenize_tasks.txt}"
if [[ ! -f "${TASKS}" ]]; then
  echo "ERROR: missing ${TASKS}" >&2
  exit 1
fi
N_TASKS=$(grep -cve '^[[:space:]]*$' "${TASKS}" || true)
if [[ "${N_TASKS}" -lt 1 ]]; then
  echo "ERROR: no tokenize tasks in ${TASKS}" >&2
  exit 1
fi
echo "tokenize_tasks=${N_TASKS} file=${TASKS}"

COMMON_EXPORT="ALL,RUN_DIR=${RUN_DIR},VENV=${VENV},PLAN=${PLAN},SOURCE_LIST=${SOURCE_LIST},SCRATCH_ROOT=${SCRATCH_ROOT},REFHQ_NEW_SCRIPTS=${REFHQ_NEW_SCRIPTS},STAGE_DIR=${STAGE_DIR},S3_BUCKET=${S3_BUCKET},S3_PREFIX=${S3_PREFIX},SEED=${SEED},TOKENIZE_TASKS=${TASKS},AWS_SESSION_ENV=${AWS_SESSION_ENV:-${RUN_DIR}/aws-session.env}"

TOK_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --array=0-$((N_TASKS - 1)) \
  --chdir="${RUN_DIR}" \
  --export="${COMMON_EXPORT}" \
  "${REFHQ_NEW_SCRIPTS}/tokenize_source.sbatch")
echo "tokenize_job=${TOK_JOB}"

MERGE_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterok:${TOK_JOB} \
  --chdir="${RUN_DIR}" \
  --export="${COMMON_EXPORT}" \
  "${REFHQ_NEW_SCRIPTS}/merge_tokenized.sbatch")
echo "merge_job=${MERGE_JOB}"

FIN_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterok:${MERGE_JOB} \
  --chdir="${RUN_DIR}" \
  --export="${COMMON_EXPORT}" \
  "${REFHQ_NEW_SCRIPTS}/finalize_upload.sbatch")
echo "finalize_job=${FIN_JOB}"

if [[ "${SKIP_PUBLISH:-0}" == "1" ]]; then
  echo "skip_publish=1"
  echo "publish_job=skipped"
else
  PUB_JOB=$(sbatch --parsable --exclude=wheat-01 \
    --dependency=afterok:${FIN_JOB} \
    --chdir="${RUN_DIR}" \
    --export="${COMMON_EXPORT}" \
    "${REFHQ_NEW_SCRIPTS}/publish_refhq_new.sbatch")
  echo "publish_job=${PUB_JOB}"
fi
