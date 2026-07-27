#!/usr/bin/env bash
# Plan + download + shard-select finalize + bulk-upload to S3.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
BASE_RUN="${BASE_RUN:-/scratch/users/${SUNET}/agent-runs/olmo-mix-30b-20260722}"
RUN_NAME="${RUN_NAME:-olmo-mix-upsample-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
CAP_TOKENS="${CAP_TOKENS:-20000000000}"
BUCKET="${BUCKET:-edullm-dataset-olmohq}"
PREFIX="${PREFIX:-olmo-mix-1124-30b}"
EDULLM_ROOT="${EDULLM_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
DOMAIN_LIST="${DOMAIN_LIST:-starcoder pes2o arxiv open-web-math algebraic-stack wiki}"

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${RUN_DIR}/data/data" "${RUN_DIR}/plan"
cd "${RUN_DIR}"

if [[ ! -d "${BASE_RUN}/data/data/dclm" ]]; then
  echo "missing DCLM data in ${BASE_RUN}/data/data/dclm" >&2
  exit 1
fi

DATASETS_SHARED="${EDULLM_ROOT}/datasets"
FARMSHARE="${EDULLM_ROOT}/scripts/farmshare"
for f in olmo_shard_utils.py download_s3_shard.py trim_olmo_overshoot.py trim_and_tokenize_regmix.py; do
  cp -a "${DATASETS_SHARED}/${f}" "${RUN_DIR}/scripts/"
done
cp -a "${DATASETS_SHARED}/download_s3_shard.sbatch" "${RUN_DIR}/scripts/"
cp -a "${FARMSHARE}/prepare_aws_session.sh" "${RUN_DIR}/scripts/" 2>/dev/null || true
cp -a "${FARMSHARE}/write_aws_session_env.py" "${RUN_DIR}/scripts/" 2>/dev/null || true

python3 -m venv "${RUN_DIR}/venv"
# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
pip install -U pip wheel
pip install "huggingface_hub[hf_transfer]" hf_transfer boto3 tqdm transformers zstandard

# Reuse DCLM shards from the prior 30B run (no re-download).
ln -sfn "${BASE_RUN}/data/data/dclm" "${RUN_DIR}/data/data/dclm"

python "${RUN_DIR}/scripts/plan_olmo_mix_upsample.py" \
  --base-run-dir "${BASE_RUN}" \
  --cap-tokens "${CAP_TOKENS}" \
  --out-dir "${RUN_DIR}/plan"

MANIFEST="${RUN_DIR}/plan/manifest.jsonl"
SUMMARY="${RUN_DIR}/plan/summary.json"

# Download only non-DCLM shards.
NON_DCLM_MANIFEST="${RUN_DIR}/plan/manifest_non_dclm.jsonl"
export RUN_DIR
python - <<'PY'
import json
from pathlib import Path
import os
run = Path(os.environ["RUN_DIR"])
rows = [json.loads(l) for l in (run / "plan/manifest.jsonl").read_text().splitlines()]
non = [r for r in rows if r.get("domain") != "dclm"]
(run / "plan/manifest_non_dclm.jsonl").write_text("\n".join(json.dumps(r) for r in non) + "\n")
print(f"non_dclm_files={len(non)}")
PY

N=$(wc -l < "${NON_DCLM_MANIFEST}")
echo "non_dclm_manifest_files=${N}"

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${RUN_DIR}/venv
MANIFEST=${MANIFEST}
NON_DCLM_MANIFEST=${NON_DCLM_MANIFEST}
SUMMARY=${SUMMARY}
LOCAL_ROOT=${RUN_DIR}/data
BUCKET=${BUCKET}
PREFIX=${PREFIX}
DOMAIN_LIST="${DOMAIN_LIST}"
EDULLM_ROOT=${EDULLM_ROOT}
EOF

# shellcheck disable=SC1091
source "${RUN_DIR}/env.sh"

if [[ "${N}" -gt 0 ]]; then
  DL_JOB=$(sbatch --parsable --exclude=wheat-01 \
    --array=0-$((N - 1))%40 \
    --chdir="${RUN_DIR}" \
    --export=ALL,RUN_DIR="${RUN_DIR}",VENV="${VENV}",LOCAL_ROOT="${LOCAL_ROOT}",MANIFEST="${NON_DCLM_MANIFEST}" \
    "${RUN_DIR}/scripts/download_olmo_shard.sbatch")
  echo "download_job_id=${DL_JOB}"
  UP_DEP="afterok:${DL_JOB}"
else
  DL_JOB=""
  echo "download_job_id=skipped"
  UP_DEP=""
fi

NDOMS=$(echo "${DOMAIN_LIST}" | wc -w)
unset NDOMS

if [[ -n "${UP_DEP}" ]]; then
  FINAL_DEP="${UP_DEP}"
else
  FINAL_DEP=""
fi

FINAL_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task=4 \
  --mem=16G \
  --time=12:00:00 \
  ${FINAL_DEP:+--dependency="${FINAL_DEP}"} \
  --chdir="${RUN_DIR}" \
  --wrap "set -Eeuo pipefail; source ${RUN_DIR}/env.sh; source ${VENV}/bin/activate; export EDULLM_ROOT=${EDULLM_ROOT}; export RUN_DIR=${RUN_DIR}; source ${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session.sh; python ${RUN_DIR}/scripts/finalize_olmo_upsample_upload.py --run-dir ${RUN_DIR} --bucket ${BUCKET} --prefix ${PREFIX}")
echo "finalize_upload_job_id=${FINAL_JOB}"

echo "RUN_DIR=${RUN_DIR}" | tee "${RUN_DIR}/RUN_DIR.txt"
cat "${SUMMARY}"
