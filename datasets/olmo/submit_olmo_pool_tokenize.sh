#!/usr/bin/env bash
# Dolma2-tokenize an OLMo-mix pool already on S3; upload tokenized/ back to the same bucket.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-}"
SRC_BUCKET="${SRC_BUCKET:?set SRC_BUCKET}"
SRC_PREFIX="${SRC_PREFIX:-olmo-mix-1124-30b}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME:-${SRC_BUCKET}-dolma2-tok-$(date +%Y%m%d-%H%M%S)}}"
EDULLM_ROOT="${EDULLM_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
LOCAL_MIRROR="${LOCAL_MIRROR:-}"
BASE_RUN_DIR="${BASE_RUN_DIR:-}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
DL_CONCURRENCY="${DL_CONCURRENCY:-60}"
TOK_CONCURRENCY="${TOK_CONCURRENCY:-40}"

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${RUN_DIR}/data" "${RUN_DIR}/plan" "${RUN_DIR}/tokenized/shards"
cd "${RUN_DIR}"

OLMO_ROOT="${EDULLM_ROOT}/datasets/olmo"
DATASETS_SHARED="${EDULLM_ROOT}/datasets"
FARMSHARE="${EDULLM_ROOT}/scripts/farmshare"
for f in download_s3_shard.py download_s3_shard.sbatch; do
  cp -a "${DATASETS_SHARED}/${f}" "${RUN_DIR}/scripts/"
done
for f in build_pool_tokenize_map.py \
  tokenize_olmo_shard.py tokenize_olmo_shard.sbatch finalize_pool_tokenized_upload.py \
  finalize_pool_tokenized_upload.sbatch retry_tokenize_missing.py retry_tokenize_missing.sbatch; do
  cp -a "${OLMO_ROOT}/${f}" "${RUN_DIR}/scripts/"
done
for f in prepare_aws_session_light.sh write_aws_session_env.py; do
  cp -a "${FARMSHARE}/${f}" "${RUN_DIR}/scripts/"
done
sed -i 's/\r$//' "${RUN_DIR}/scripts/"*.{sh,sbatch,py} 2>/dev/null || true

if [[ -n "${BASE_RUN_DIR}" && -f "${BASE_RUN_DIR}/.hf_token" ]]; then
  cp -a "${BASE_RUN_DIR}/.hf_token" "${RUN_DIR}/.hf_token"
fi
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${RUN_DIR}/.hf_token}"

if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
fi
# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
pip install -U pip wheel
pip install boto3 tqdm transformers "numpy<2.1" zstandard sentencepiece protobuf

export EDULLM_ROOT RUN_DIR
export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"
if [[ -n "${AWS_SESSION_ENV:-}" && -f "${AWS_SESSION_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${AWS_SESSION_ENV}"
  aws sts get-caller-identity --output text >/dev/null
  cp -a "${AWS_SESSION_ENV}" "${RUN_DIR}/aws-session.env"
  AWS_SESSION_ENV="${RUN_DIR}/aws-session.env"
  echo "aws_session_ready preset=${AWS_SESSION_ENV}"
else
  # shellcheck disable=SC1091
  source "${RUN_DIR}/scripts/prepare_aws_session_light.sh"
  # shellcheck disable=SC1090
  source "${AWS_SESSION_ENV}"
  cp -a "${AWS_SESSION_ENV}" "${RUN_DIR}/aws-session.env"
  AWS_SESSION_ENV="${RUN_DIR}/aws-session.env"
fi

MANIFEST="${RUN_DIR}/plan/manifest.jsonl"
SUMMARY="${RUN_DIR}/plan/summary.json"
aws s3 cp "s3://${SRC_BUCKET}/${SRC_PREFIX}/plan/manifest.jsonl" "${MANIFEST}"
aws s3 cp "s3://${SRC_BUCKET}/${SRC_PREFIX}/plan/summary.json" "${SUMMARY}" || true
N=$(wc -l < "${MANIFEST}")

if [[ -n "${BASE_RUN_DIR}" && -d "${BASE_RUN_DIR}" ]]; then
  echo "staging local base run ${BASE_RUN_DIR}"
  for domain in algebraic-stack arxiv open-web-math pes2o starcoder wiki; do
    for suffix in trimmed upsampled; do
      src="${BASE_RUN_DIR}/trim/${domain}/${domain}-${suffix}.json.gz"
      if [[ ! -f "${src}" ]]; then
        src="${BASE_RUN_DIR}/data/data/${domain}/${domain}-${suffix}.json.gz"
      fi
      if [[ -f "${src}" ]]; then
        dst="${RUN_DIR}/data/data/${domain}/${domain}-${suffix}.json.gz"
        mkdir -p "$(dirname "${dst}")"
        if [[ ! -e "${dst}" ]]; then
          ln -s "${src}" "${dst}" 2>/dev/null || cp -a "${src}" "${dst}"
        fi
      fi
    done
  done
  if [[ -d "${BASE_RUN_DIR}/data/data/dclm" ]]; then
    find "${BASE_RUN_DIR}/data/data/dclm" -type f \( -name '*.zstd' -o -name '*.json.gz' -o -name '*.jsonl.zstd' \) | while read -r src; do
      rel="${src#${BASE_RUN_DIR}/data/}"
      dst="${RUN_DIR}/data/${rel}"
      mkdir -p "$(dirname "${dst}")"
      if [[ ! -e "${dst}" ]]; then
        ln -s "${src}" "${dst}" 2>/dev/null || cp -a "${src}" "${dst}"
      fi
    done
  fi
fi

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${RUN_DIR}/venv
MANIFEST=${MANIFEST}
SUMMARY=${SUMMARY}
LOCAL_ROOT=${RUN_DIR}/data
SRC_BUCKET=${SRC_BUCKET}
SRC_PREFIX=${SRC_PREFIX}
AWS_SESSION_ENV=${AWS_SESSION_ENV}
HF_TOKEN_FILE=${HF_TOKEN_FILE}
EOF

SBATCH_EXPORT_COMMON="RUN_DIR=${RUN_DIR},VENV=${RUN_DIR}/venv,MANIFEST=${MANIFEST},SRC_BUCKET=${SRC_BUCKET},SRC_PREFIX=${SRC_PREFIX},AWS_SESSION_ENV=${AWS_SESSION_ENV},HF_TOKEN_FILE=${HF_TOKEN_FILE}"

DL_JOB=""
if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
  echo "skip_download=1"
else
  DL_JOB=$(sbatch --parsable --exclude=wheat-01 \
    --array=0-$((N - 1))%${DL_CONCURRENCY} \
    --chdir="${RUN_DIR}" \
    --export=ALL,${SBATCH_EXPORT_COMMON},LOCAL_ROOT="${RUN_DIR}/data",LOCAL_MIRROR="${LOCAL_MIRROR}" \
    "${RUN_DIR}/scripts/download_s3_shard.sbatch")
  echo "download_job=${DL_JOB}"
fi

TRIM_ARGS=()
if [[ -n "${BASE_RUN_DIR}" && -d "${BASE_RUN_DIR}/trim" ]]; then
  TRIM_ARGS=(--trim-root "${BASE_RUN_DIR}/trim")
fi

MAP_DEP=""
if [[ -n "${DL_JOB}" ]]; then
  MAP_DEP="--dependency=afterok:${DL_JOB}"
fi

MAP_JOB=$(sbatch --parsable --exclude=wheat-01 \
  ${MAP_DEP} \
  --cpus-per-task=4 \
  --mem=8G \
  --time=01:00:00 \
  --job-name=pool-map \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/map-%j.out" \
  --error="${RUN_DIR}/logs/map-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; source ${RUN_DIR}/venv/bin/activate; python ${RUN_DIR}/scripts/build_pool_tokenize_map.py --manifest ${MANIFEST} --data-root ${RUN_DIR}/data ${TRIM_ARGS[*]} --out-dir ${RUN_DIR}/tokenized --map-file ${RUN_DIR}/tokenize_map.txt'")
echo "map_job=${MAP_JOB}"

mkdir -p "${RUN_DIR}/hf-cache"

TOK_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterok:${MAP_JOB} \
  --array=0-$((N - 1))%${TOK_CONCURRENCY} \
  --chdir="${RUN_DIR}" \
  --export=ALL,${SBATCH_EXPORT_COMMON} \
  --output="${RUN_DIR}/logs/tokenize_%A_%a.out" \
  --error="${RUN_DIR}/logs/tokenize_%A_%a.err" \
  "${RUN_DIR}/scripts/tokenize_olmo_shard.sbatch")
echo "tokenize_job=${TOK_JOB}"

RETRY_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterany:${TOK_JOB} \
  --chdir="${RUN_DIR}" \
  --export=ALL,${SBATCH_EXPORT_COMMON} \
  "${RUN_DIR}/scripts/retry_tokenize_missing.sbatch")
echo "retry_job=${RETRY_JOB}"

UP_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterok:${RETRY_JOB} \
  --chdir="${RUN_DIR}" \
  --export=ALL,${SBATCH_EXPORT_COMMON} \
  "${RUN_DIR}/scripts/finalize_pool_tokenized_upload.sbatch")
echo "upload_job=${UP_JOB}"

echo "RUN_DIR=${RUN_DIR}"
echo "manifest_shards=${N}"
echo "s3://${SRC_BUCKET}/${SRC_PREFIX}/tokenized/"
