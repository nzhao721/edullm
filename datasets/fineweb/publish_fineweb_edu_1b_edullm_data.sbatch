#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=fw1b-edullm-pub
#SBATCH --output=logs/fineweb_edu_1b_publish_%j.out
#SBATCH --error=logs/fineweb_edu_1b_publish_%j.err

# Publish FineWeb-Edu 1B: tokenizer/smollm2 + pretrain/fineweb-edu-1b (tokens+text).
set -Eeuo pipefail

: "${RUN_DIR:?}" "${STAGE_DIR:?}" "${EDULLM_ROOT:?}" "${AWS_SESSION_ENV:?}"

TOKENIZED_ROOT="${TOKENIZED_ROOT:-/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-tokenized}"
RAW_ROOT="${RAW_ROOT:-/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-raw}"
TOKENIZER_SNAPSHOT="${TOKENIZER_SNAPSHOT:-/scratch/users/nzhao2/hf-cache/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2}"
TOKENIZER_STAGE_DIR="${TOKENIZER_STAGE_DIR:-${RUN_DIR}/tokenizer-stage}"
VENV="${VENV:-${RUN_DIR}/venv}"
EDULLM_DATA_DIR="${EDULLM_DATA_DIR:-${RUN_DIR}/edullm-data}"
SKIP_EDULLM_DATA_INSTALL="${SKIP_EDULLM_DATA_INSTALL:-0}"

mkdir -p "${RUN_DIR}/logs"
cd "${RUN_DIR}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV}"

export PATH="${PATH}:${HOME}/.local/bin:${HOME}/tools/aws/bin"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
unset AWS_PROFILE || true

pip install -q -U pip wheel
pip install -q boto3 numpy

if [[ "${SKIP_EDULLM_DATA_INSTALL:-0}" != 1 ]]; then
  rm -rf "${EDULLM_DATA_DIR}"
  git clone --depth 1 --branch v0.2.0 https://github.com/edu-llm/edullm-data.git "${EDULLM_DATA_DIR}"
  pip install -q -e "${EDULLM_DATA_DIR}"
fi

python -c 'import edullm_data, boto3; print("edullm_data ok", edullm_data.__file__); print(boto3.client("sts", region_name="us-east-1").get_caller_identity()["Arn"])'

EXTRA_ARGS=()
if [[ "${SKIP_STAGE:-0}" == 1 ]]; then
  EXTRA_ARGS+=(--skip-stage)
fi
if [[ "${SKIP_TOKENIZER_PUBLISH:-0}" == 1 ]]; then
  EXTRA_ARGS+=(--skip-tokenizer-publish)
fi

python "${EDULLM_ROOT}/datasets/fineweb/publish_fineweb_edu_1b_edullm_data.py" \
  --tokenized-root "${TOKENIZED_ROOT}" \
  --raw-root "${RAW_ROOT}" \
  --tokenizer-snapshot "${TOKENIZER_SNAPSHOT}" \
  --stage-dir "${STAGE_DIR}" \
  --tokenizer-stage-dir "${TOKENIZER_STAGE_DIR}" \
  --shard-bytes 1073741824 \
  --val-fraction 0.0015 \
  --hash-workers 8 \
  --copy-workers 8 \
  --force \
  "${EXTRA_ARGS[@]}"
