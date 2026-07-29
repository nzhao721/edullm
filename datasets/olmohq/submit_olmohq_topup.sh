#!/usr/bin/env bash
# Append-only top-up of olmohq starcoder + pes2o so |plan-meas|/meas <= 10%.
# NEVER modifies regmix-10b. Only appends under s3://edullm-datasets/olmo100b/olmo-mix-1124-30b/.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-olmohq-topup-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
EDULLM_ROOT="${EDULLM_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
BUCKET="${BUCKET:-edullm-datasets}"
OLMOHQ_PREFIX="${OLMOHQ_PREFIX:-olmo100b/olmo-mix-1124-30b}"
BASE_RUN="${BASE_RUN:-/scratch/users/${SUNET}/agent-runs/olmo-mix-30b-20260722}"
DL_CONCURRENCY="${DL_CONCURRENCY:-40}"
TOK_CONCURRENCY="${TOK_CONCURRENCY:-20}"

# nvm (used by AWS session mint) rejects a set PREFIX env var.
unset PREFIX || true
export OLMOHQ_PREFIX

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${RUN_DIR}/data" "${RUN_DIR}/plan" "${RUN_DIR}/tokenized/shards"
cd "${RUN_DIR}"

OLMOHQ="${EDULLM_ROOT}/datasets/olmohq"
OLMO="${EDULLM_ROOT}/datasets/olmo"
DATASETS="${EDULLM_ROOT}/datasets"
FARMSHARE="${EDULLM_ROOT}/scripts/farmshare"

cp -a "${OLMOHQ}/plan_olmohq_topup.py" "${RUN_DIR}/scripts/"
cp -a "${OLMOHQ}/finalize_olmohq_topup_upload.py" "${RUN_DIR}/scripts/"
cp -a "${DATASETS}/olmo_shard_utils.py" "${RUN_DIR}/scripts/"
cp -a "${OLMO}/download_olmo_shard.py" "${RUN_DIR}/scripts/"
cp -a "${OLMO}/tokenize_olmo_shard.py" "${RUN_DIR}/scripts/"
cp -a "${OLMO}/build_pool_tokenize_map.py" "${RUN_DIR}/scripts/"
cp -a "${FARMSHARE}/prepare_aws_session_light.sh" "${RUN_DIR}/scripts/"
cp -a "${FARMSHARE}/write_aws_session_env.py" "${RUN_DIR}/scripts/"
sed -i 's/\r$//' "${RUN_DIR}/scripts/"*.{sh,sbatch,py} 2>/dev/null || true

if [[ -f "${BASE_RUN}/.hf_token" ]]; then
  cp -a "${BASE_RUN}/.hf_token" "${RUN_DIR}/.hf_token"
fi

if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
fi
# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
pip install -U pip wheel
pip install "huggingface_hub[hf_transfer]" hf_transfer boto3 tqdm transformers "numpy<2.1" zstandard sentencepiece protobuf

export EDULLM_ROOT RUN_DIR
# shellcheck disable=SC1091
source "${RUN_DIR}/scripts/prepare_aws_session_light.sh"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV}"

aws s3 cp "s3://${BUCKET}/${OLMOHQ_PREFIX}/plan/tokenized_manifest.json" "${RUN_DIR}/plan/tokenized_manifest.json"
aws s3 cp "s3://${BUCKET}/${OLMOHQ_PREFIX}/plan/manifest.jsonl" "${RUN_DIR}/plan/manifest.jsonl"

HF_ARGS=()
if [[ -f "${RUN_DIR}/.hf_token" ]]; then
  HF_ARGS=(--token "$(tr -d '\r\n' < "${RUN_DIR}/.hf_token")")
fi

PY="${RUN_DIR}/venv/bin/python"
"$PY" "${RUN_DIR}/scripts/plan_olmohq_topup.py" \
  --tokenized-manifest "${RUN_DIR}/plan/tokenized_manifest.json" \
  --pool-manifest "${RUN_DIR}/plan/manifest.jsonl" \
  --out-dir "${RUN_DIR}/plan" \
  --domains starcoder pes2o \
  "${HF_ARGS[@]}"

N=$(wc -l < "${RUN_DIR}/plan/topup_manifest.jsonl" | tr -d ' ')
echo "topup_files=${N}"
cat "${RUN_DIR}/plan/topup_summary.json"
if [[ "${N}" -eq 0 ]]; then
  echo "nothing to top up"
  exit 0
fi

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${RUN_DIR}/venv
EDULLM_ROOT=${EDULLM_ROOT}
AWS_SESSION_ENV=${AWS_SESSION_ENV}
BUCKET=${BUCKET}
OLMOHQ_PREFIX=${OLMOHQ_PREFIX}
N=${N}
EOF

# --- download array ---
cat > "${RUN_DIR}/scripts/download_topup.sbatch" <<'EOF'
#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --exclude=wheat-01
set -Eeuo pipefail
unset PREFIX || true
source "${RUN_DIR}/env.sh"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
if [[ -f "${RUN_DIR}/.hf_token" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '\r\n' < "${RUN_DIR}/.hf_token")"
fi
export HF_HUB_ENABLE_HF_TRANSFER=1
python "${RUN_DIR}/scripts/download_olmo_shard.py" \
  --manifest "${RUN_DIR}/plan/topup_manifest.jsonl" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --local-root "${RUN_DIR}/data"
EOF

DL_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --array=0-$((N - 1))%"${DL_CONCURRENCY}" \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/dl-%A_%a.out" \
  --error="${RUN_DIR}/logs/dl-%A_%a.err" \
  --export=ALL,RUN_DIR,VENV \
  "${RUN_DIR}/scripts/download_topup.sbatch")
echo "download_job_id=${DL_JOB}"
echo "${DL_JOB}" > "${RUN_DIR}/download_job_id.txt"

# --- tokenize map + array ---
cat > "${RUN_DIR}/scripts/build_topup_tokenize_map.py" <<'PY'
import json, hashlib
from pathlib import Path
import os
run = Path(os.environ["RUN_DIR"])
rows = [json.loads(l) for l in (run/"plan/topup_manifest.jsonl").read_text().splitlines() if l.strip()]
lines = []
index = []
for i, row in enumerate(rows):
    rel = row["path"]
    domain = row["domain"]
    src = run / "data" / rel
    stem = Path(rel).name
    for suf in (".jsonl.zstd", ".jsonl.zst", ".json.gz", ".jsonl.gz", ".zstd", ".jsonl", ".json"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    h = hashlib.sha1(rel.encode()).hexdigest()[:10]
    out_rel = f"shards/topup_{i:05d}__{domain}__{stem[:60]}__{h}.npy"
    out = run / "tokenized" / out_rel
    lines.append(f"{src}|{out}")
    index.append({"index": i, "manifest_path": rel, "domain": domain, "npy": out_rel, "input": str(src)})
(run/"tokenize_map.txt").write_text("\n".join(lines) + "\n")
(run/"plan/topup_tokenize_index.jsonl").write_text("\n".join(json.dumps(r) for r in index) + "\n")
print(f"map_lines={len(lines)}")
PY

cat > "${RUN_DIR}/scripts/tokenize_topup.sbatch" <<'EOF'
#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=wheat-01
set -Eeuo pipefail
unset PREFIX || true
source "${RUN_DIR}/env.sh"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${RUN_DIR}/tokenize_map.txt")
SRC="${LINE%%|*}"
DST="${LINE##*|}"
mkdir -p "$(dirname "${DST}")"
python "${RUN_DIR}/scripts/tokenize_olmo_shard.py" --input "${SRC}" --output "${DST}"
EOF

# Slurm --wrap runs under /bin/sh; always use bash -lc. Put aws CLI on PATH.
_AWS_PATH='export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"'

# Build map after downloads complete, then tokenize.
MAP_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal --cpus-per-task=2 --mem=4G --time=00:30:00 \
  --dependency="afterok:${DL_JOB}" \
  --job-name=topup-map \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/map-%j.out" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; source ${RUN_DIR}/env.sh; source ${RUN_DIR}/venv/bin/activate; export RUN_DIR=${RUN_DIR}; python ${RUN_DIR}/scripts/build_topup_tokenize_map.py'")
echo "map_job_id=${MAP_JOB}"

TOK_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --array=0-$((N - 1))%"${TOK_CONCURRENCY}" \
  --dependency="afterok:${MAP_JOB}" \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/tok-%A_%a.out" \
  --error="${RUN_DIR}/logs/tok-%A_%a.err" \
  --export=ALL,RUN_DIR \
  "${RUN_DIR}/scripts/tokenize_topup.sbatch")
echo "tokenize_job_id=${TOK_JOB}"
echo "${TOK_JOB}" > "${RUN_DIR}/tokenize_job_id.txt"

# --- finalize append upload (login-minted aws-session.env only; never remint on compute) ---
UP_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal --cpus-per-task=8 --mem=32G --time=12:00:00 \
  --dependency="afterok:${TOK_JOB}" \
  --job-name=topup-up \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/upload-%j.out" \
  --error="${RUN_DIR}/logs/upload-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; ${_AWS_PATH}; source ${RUN_DIR}/env.sh; source \${AWS_SESSION_ENV}; source ${RUN_DIR}/venv/bin/activate; python ${RUN_DIR}/scripts/finalize_olmohq_topup_upload.py --run-dir ${RUN_DIR} --bucket ${BUCKET} --prefix \${OLMOHQ_PREFIX}'")
echo "upload_job_id=${UP_JOB}"
echo "${UP_JOB}" > "${RUN_DIR}/upload_job_id.txt"

echo "RUN_DIR=${RUN_DIR}"
echo "submitted dl=${DL_JOB} map=${MAP_JOB} tok=${TOK_JOB} up=${UP_JOB}"
