#!/usr/bin/env bash
# Orchestrate refhq-new FarmShare pipeline -> s3://edullm-datasets/refhq/refhq-new/
# then publish() as pretrain/refhq-new (tokenizer/dolma2-bpe).
#
# Dependency chain (afterok):
#   plan → download[0-5] → normalize[0-5] → dolma eng[0-5] → holdout
#        → tokenize[0..N-1 from tokenize_tasks.txt] → finalize → publish
#
# Sibling contracts (datasets/refhq_new/scripts/):
#   Env: RUN_DIR VENV PLAN SOURCE_LIST REFHQ_NEW_SCRIPTS (+ AWS_SESSION_ENV late)
#   Download/normalize/english: Slurm array 0-5 over SOURCE_LIST (no % throttle)
#   Holdout: holdout_docs.sbatch refreshes manifests/tokenize_tasks.txt
#   Tokenize: array over tokenize_tasks.txt lines "source domain split"
#     (SLURM_ARRAY_TASK_ID = line index; tokenize_source.py --task-index)
#     emits tokenized/<source>/<domain>/{train,val}.npy (+ .json)
#     finalize also accepts tok/ alias of tokenized/
#   Docs layout:
#     docs/<source>/<domain>/documents-*.jsonl.gz
#     out/<source>/<domain>/documents/…  # after Dolma English (/documents/ required)
#     holdout/<source>/<domain>/{train,val}/documents-*.jsonl.gz
#   Publish layout (no token carve; doc holdout already done):
#     tokens/<source>/<domain>/{train,val}-NNNNN.u32le.bin
#
# AWS: mint on laptop, push session — never sb-aws-creds login on FarmShare:
#   scripts/farmshare/push_aws_session_to_farmshare.sh "${RUN_DIR}"
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-refhq-new-v1}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/users/${SUNET}/${RUN_NAME}}"
RUN_DIR="${RUN_DIR:-${SCRATCH_ROOT}}"
STAGING_ROOT="${STAGING_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
SOURCE_LIST="${SOURCE_LIST:-tulu-v2 openhermes-25 tulu-3 hermes-3 smoltalk dolci}"
SEED="${SEED:-42}"
S3_BUCKET="${S3_BUCKET:-edullm-datasets}"
S3_PREFIX="${S3_PREFIX:-refhq/refhq-new}"
STAGE_DIR="${STAGE_DIR:-${RUN_DIR}/publish-stage}"
REFHQ_NEW_SCRIPTS="${REFHQ_NEW_SCRIPTS:-${RUN_DIR}/datasets/refhq_new/scripts}"
SKIP_PUBLISH="${SKIP_PUBLISH:-0}"
HF_TOKEN_SRC="${HF_TOKEN_SRC:-/scratch/users/${SUNET}/hq-reference-v1/.hf_token}"

# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/manifests" "${RUN_DIR}/raw" "${RUN_DIR}/tokenized"
cd "${RUN_DIR}"

if [[ -d "${STAGING_ROOT}/datasets/refhq_new" ]]; then
  refhq_new_sync_to_run "${STAGING_ROOT}" "${RUN_DIR}"
fi
refhq_new_export_pythonpath "${RUN_DIR}"
REFHQ_NEW_SCRIPTS="${RUN_DIR}/datasets/refhq_new/scripts"

# Prefer copying HF token before any download array starts (Hermes-3 is gated).
if [[ ! -f "${RUN_DIR}/.hf_token" ]]; then
  for cand in "${HF_TOKEN_SRC}" \
    "/scratch/users/${SUNET}/hq-reference-v1/.hf_token" \
    "/scratch/users/${SUNET}/refhq-regmix-5p5b-v1/.hf_token"
  do
    if [[ -n "${cand}" && -f "${cand}" ]]; then
      cp -a "${cand}" "${RUN_DIR}/.hf_token"
      chmod 600 "${RUN_DIR}/.hf_token"
      echo "copied HF token from ${cand}"
      break
    fi
  done
fi
if [[ ! -f "${RUN_DIR}/.hf_token" ]]; then
  echo "ERROR: missing ${RUN_DIR}/.hf_token (required for gated Hermes-3)" >&2
  exit 1
fi

if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
  # shellcheck disable=SC1091
  source "${RUN_DIR}/venv/bin/activate"
  pip install -U pip wheel
  # dolma 1.1.2 pins tokenizers<=0.19.1 — install a compatible transformers stack together
  pip install \
    "huggingface_hub[hf_transfer]>=0.23,<1" hf_transfer \
    "datasets>=2.19,<3" "tokenizers>=0.15.0,<=0.19.1" "transformers>=4.40,<4.45" \
    tqdm zstandard "numpy<2" \
    "dolma[code]==1.1.2" pyyaml boto3 awscli
else
  # shellcheck disable=SC1091
  source "${RUN_DIR}/venv/bin/activate"
fi

# --- plan (local) ---
read -r -a SOURCES_ARR <<< "${SOURCE_LIST}"
python "${REFHQ_NEW_SCRIPTS}/plan_refhq_new.py" \
  --scratch-root "${SCRATCH_ROOT}" \
  --seed "${SEED}" \
  --s3-bucket "${S3_BUCKET}" \
  --s3-prefix "${S3_PREFIX}" \
  --sources "${SOURCES_ARR[@]}"

PLAN="${PLAN:-${SCRATCH_ROOT}/manifests/plan.json}"
: "${PLAN:?plan.json missing}"

N=${#SOURCES_ARR[@]}

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${RUN_DIR}/venv
PLAN=${PLAN}
SOURCE_LIST="${SOURCE_LIST}"
SCRATCH_ROOT=${SCRATCH_ROOT}
STAGING_ROOT=${STAGING_ROOT}
S3_BUCKET=${S3_BUCKET}
S3_PREFIX=${S3_PREFIX}
STAGE_DIR=${STAGE_DIR}
REFHQ_NEW_SCRIPTS=${REFHQ_NEW_SCRIPTS}
SEED=${SEED}
TOKENIZE_TASKS=${SCRATCH_ROOT}/manifests/tokenize_tasks.txt
ENGLISH_TASKS=${SCRATCH_ROOT}/manifests/english_tasks.txt
EOF

# shellcheck disable=SC1091
source "${RUN_DIR}/env.sh"
VENV="${RUN_DIR}/venv"
refhq_new_export_pythonpath "${RUN_DIR}"

AWS_SESSION_ENV="${AWS_SESSION_ENV:-${RUN_DIR}/aws-session.env}"
if [[ ! -f "${AWS_SESSION_ENV}" ]]; then
  echo "WARN: ${AWS_SESSION_ENV} missing." >&2
  echo "  From the engineer laptop (not FarmShare):" >&2
  echo "  scripts/farmshare/push_aws_session_to_farmshare.sh ${RUN_DIR}" >&2
  echo "  finalize/publish need the session; push before those jobs start." >&2
fi

COMMON_EXPORT="ALL,RUN_DIR=${RUN_DIR},VENV=${VENV},PLAN=${PLAN},SOURCE_LIST=${SOURCE_LIST},SCRATCH_ROOT=${SCRATCH_ROOT},REFHQ_NEW_SCRIPTS=${REFHQ_NEW_SCRIPTS},STAGE_DIR=${STAGE_DIR},S3_BUCKET=${S3_BUCKET},S3_PREFIX=${S3_PREFIX},SEED=${SEED},TOKENIZE_TASKS=${TOKENIZE_TASKS},ENGLISH_TASKS=${ENGLISH_TASKS},AWS_SESSION_ENV=${AWS_SESSION_ENV}"

require_sbatch() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing ${path}" >&2
    exit 1
  fi
}

require_sbatch "${REFHQ_NEW_SCRIPTS}/download_hf_source.sbatch"
require_sbatch "${REFHQ_NEW_SCRIPTS}/normalize_filter_source.sbatch"
require_sbatch "${REFHQ_NEW_SCRIPTS}/dolma_english_filter.sbatch"
require_sbatch "${REFHQ_NEW_SCRIPTS}/holdout_docs.sbatch"
require_sbatch "${REFHQ_NEW_SCRIPTS}/tokenize_source.sbatch"
require_sbatch "${REFHQ_NEW_SCRIPTS}/finalize_upload.sbatch"
require_sbatch "${REFHQ_NEW_SCRIPTS}/publish_refhq_new.sbatch"

# 1) download array 0-(N-1), no % throttle
DL_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --array=0-$((N - 1)) \
  --chdir="${RUN_DIR}" \
  --export="${COMMON_EXPORT}" \
  "${REFHQ_NEW_SCRIPTS}/download_hf_source.sbatch")
echo "download_job=${DL_JOB}"

# 2) normalize + metadata filter
NORM_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterok:${DL_JOB} \
  --array=0-$((N - 1)) \
  --chdir="${RUN_DIR}" \
  --export="${COMMON_EXPORT}" \
  "${REFHQ_NEW_SCRIPTS}/normalize_filter_source.sbatch")
echo "normalize_job=${NORM_JOB}"

# 3) After normalize: build per-shard english task list, then array over shards
cat > "${RUN_DIR}/submit_post_normalize.sh" <<'POSTNORM'
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
POSTNORM
chmod +x "${RUN_DIR}/submit_post_normalize.sh"

# Post-holdout: shard-level tokenize → merge parts → finalize → publish
cat > "${RUN_DIR}/submit_post_holdout.sh" <<'POST'
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
POST
chmod +x "${RUN_DIR}/submit_post_holdout.sh"

POSTNORM_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterok:${NORM_JOB} \
  --partition=normal --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:20:00 \
  --job-name=refhqn-postnorm \
  --output=logs/refhqn_post_normalize_%j.out \
  --error=logs/refhqn_post_normalize_%j.err \
  --chdir="${RUN_DIR}" \
  --export="${COMMON_EXPORT},SKIP_PUBLISH=${SKIP_PUBLISH}" \
  --wrap="bash ${RUN_DIR}/submit_post_normalize.sh")
echo "post_normalize_job=${POSTNORM_JOB}"

echo "submitted refhq-new under ${SCRATCH_ROOT}"
echo "s3://${S3_BUCKET}/${S3_PREFIX}/"
echo "dataset_id=pretrain/refhq-instruct tokenizer=tokenizer/dolma2-bpe"
echo "chain download=${DL_JOB} normalize=${NORM_JOB} post_normalize=${POSTNORM_JOB}"
echo "AWS: scripts/farmshare/push_aws_session_to_farmshare.sh ${RUN_DIR}"
