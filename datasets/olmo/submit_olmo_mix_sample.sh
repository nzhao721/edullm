#!/usr/bin/env bash
# Bootstrap FarmShare run dir, venv, plan, and submit download array.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-olmo-mix-30b-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
TARGET_TOKENS="${TARGET_TOKENS:-30000000000}"
SEED="${SEED:-42}"
BUCKET="${BUCKET:-edullm-datasets}"
PREFIX="${PREFIX:-olmo30b/olmo-mix-1124-30b}"

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${RUN_DIR}/data" "${RUN_DIR}/plan"
cd "${RUN_DIR}"

python3 -m venv "${RUN_DIR}/venv"
# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
pip install -U pip wheel
pip install "huggingface_hub[hf_transfer]" hf_transfer boto3 tqdm

# Scripts are expected to already be copied into ${RUN_DIR}/scripts by the submitter.
python "${RUN_DIR}/scripts/plan_olmo_mix_sample.py" \
  --target-tokens "${TARGET_TOKENS}" \
  --seed "${SEED}" \
  --out-dir "${RUN_DIR}/plan"

N=$(wc -l < "${RUN_DIR}/plan/manifest.jsonl")
echo "manifest_files=${N}"
echo "RUN_DIR=${RUN_DIR}" > "${RUN_DIR}/env.sh"
echo "VENV=${RUN_DIR}/venv" >> "${RUN_DIR}/env.sh"
echo "MANIFEST=${RUN_DIR}/plan/manifest.jsonl" >> "${RUN_DIR}/env.sh"
echo "LOCAL_ROOT=${RUN_DIR}/data" >> "${RUN_DIR}/env.sh"
echo "BUCKET=${BUCKET}" >> "${RUN_DIR}/env.sh"
echo "PREFIX=${PREFIX}" >> "${RUN_DIR}/env.sh"
echo "N=${N}" >> "${RUN_DIR}/env.sh"

# shellcheck disable=SC1091
source "${RUN_DIR}/env.sh"
JOB_ID=$(sbatch --parsable --exclude=wheat-01 \
  --array=0-$((N - 1))%40 \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR,VENV,MANIFEST,LOCAL_ROOT \
  "${RUN_DIR}/scripts/download_olmo_shard.sbatch")
echo "download_job_id=${JOB_ID}"
echo "${JOB_ID}" > "${RUN_DIR}/download_job_id.txt"
cat "${RUN_DIR}/plan/summary.json"
