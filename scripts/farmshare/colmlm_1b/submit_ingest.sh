#!/usr/bin/env bash
# Submit parallel ingest: entries.db + sample-100BT array downloads.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV="${VENV:-/scratch/users/${SUNET}/agent-runs/venvs/colmlm-1b}"
HF_HOME="${HF_HOME:-/scratch/users/${SUNET}/.cache/huggingface}"
DATA_DIR="${DATA_DIR:-${RUN_DIR}/data}"
S100_DIR="${S100_DIR:-${DATA_DIR}/s100}"
ENTRIES_DB="${ENTRIES_DB:-${DATA_DIR}/fineweb_with_fullwiki_entries.db}"
NUM_S100_SHARDS="${NUM_S100_SHARDS:-35}"
EXCLUDE="${EXCLUDE:-wheat-01}"

mkdir -p "${RUN_DIR}/logs" "${S100_DIR}" "${DATA_DIR}" "${HF_HOME}"

cat > "${RUN_DIR}/env.sh" <<EOF
export RUN_DIR=${RUN_DIR}
export VENV=${VENV}
export HF_HOME=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}/hub
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_DISABLE_IMPLICIT_TOKEN=0
export PYTHONUNBUFFERED=1
export SCRIPT_DIR=${SCRIPT_DIR}
export DATA_DIR=${DATA_DIR}
export S100_DIR=${S100_DIR}
export ENTRIES_DB=${ENTRIES_DB}
export NUM_S100_SHARDS=${NUM_S100_SHARDS}
if [[ -f ${RUN_DIR}/hf_token ]]; then
  export HF_TOKEN="\$(tr -d '[:space:]' < ${RUN_DIR}/hf_token)"
  export HUGGING_FACE_HUB_TOKEN="\$HF_TOKEN"
elif [[ -f ${HF_HOME}/token ]]; then
  export HF_TOKEN="\$(tr -d '[:space:]' < ${HF_HOME}/token)"
  export HUGGING_FACE_HUB_TOKEN="\$HF_TOKEN"
fi
EOF

ENTRIES_JOB=$(sbatch --parsable --exclude="${EXCLUDE}" \
  --partition=normal \
  --cpus-per-task=4 \
  --mem=16G \
  --time=18:00:00 \
  --job-name=colmlm-entries \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/entries-%j.out" \
  --error="${RUN_DIR}/logs/entries-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; source ${RUN_DIR}/env.sh; source \${VENV}/bin/activate; python -u \${SCRIPT_DIR}/fetch_entries_db.py --out \${ENTRIES_DB}'")

S100_JOB=$(sbatch --parsable --exclude="${EXCLUDE}" \
  --partition=normal \
  --array=0-$((NUM_S100_SHARDS - 1))%35 \
  --cpus-per-task=2 \
  --mem=4G \
  --time=10:00:00 \
  --job-name=colmlm-s100 \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/s100-%A_%a.out" \
  --error="${RUN_DIR}/logs/s100-%A_%a.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; source ${RUN_DIR}/env.sh; source \${VENV}/bin/activate; python -u \${SCRIPT_DIR}/fetch_sample100bt.py --out \${S100_DIR} --shard-index \${SLURM_ARRAY_TASK_ID} --num-shards \${NUM_S100_SHARDS} --retries 14 --backoff 15'")

echo "entries_job=${ENTRIES_JOB}"
echo "s100_job=${S100_JOB}"
echo "${ENTRIES_JOB}" > "${RUN_DIR}/job_entries.txt"
echo "${S100_JOB}" > "${RUN_DIR}/job_s100.txt"
echo "RUN_DIR=${RUN_DIR}"
