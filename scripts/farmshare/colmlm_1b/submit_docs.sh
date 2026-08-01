#!/usr/bin/env bash
# Pass B only: filter sample-100BT → docs_sample.parquet (no entries.db needed).
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV="${VENV:-/scratch/users/${SUNET}/agent-runs/venvs/colmlm-1b}"
DATA_DIR="${DATA_DIR:-${RUN_DIR}/data}"
S100_DIR="${S100_DIR:-${DATA_DIR}/s100}"
DOCS_PARQUET="${DOCS_PARQUET:-${DATA_DIR}/docs_sample.parquet}"
EXCLUDE="${EXCLUDE:-wheat-01}"
MODULUS="${MODULUS:-100}"
RESIDUE="${RESIDUE:-0}"
DOCS_CPUS="${DOCS_CPUS:-32}"
DOCS_MEM="${DOCS_MEM:-96G}"

mkdir -p "${RUN_DIR}/logs" "${DATA_DIR}"

# Idempotent: skip if already submitted and still alive / completed ok
if [[ -f "${RUN_DIR}/job_docs.txt" ]]; then
  old=$(cat "${RUN_DIR}/job_docs.txt")
  if squeue -j "${old}" -h >/dev/null 2>&1; then
    echo "docs_job already running: ${old}"
    exit 0
  fi
  if [[ -f "${DOCS_PARQUET}" && -f "${DATA_DIR}/docs_stats.json" ]]; then
    echo "docs already produced; skip"
    exit 0
  fi
fi

n_parq=$(find "${S100_DIR}" -name '*.parquet' | wc -l)
if [[ "${n_parq}" -lt 140 ]]; then
  echo "ERROR: expected 140 parquet under ${S100_DIR}, found ${n_parq}" >&2
  exit 2
fi

DOCS_JOB=$(sbatch --parsable --exclude="${EXCLUDE}" \
  --partition=normal \
  --cpus-per-task="${DOCS_CPUS}" \
  --mem="${DOCS_MEM}" \
  --time=2:00:00 \
  --job-name=colmlm-docs \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/docs-%j.out" \
  --error="${RUN_DIR}/logs/docs-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; source ${RUN_DIR}/env.sh; source ${VENV}/bin/activate; python -u ${SCRIPT_DIR}/select_docs.py --parquet-glob \"${S100_DIR}/*.parquet\" --out ${DOCS_PARQUET} --work-db ${DATA_DIR}/work_docs.duckdb --threads ${DOCS_CPUS} --modulus ${MODULUS} --residue ${RESIDUE} --stats-out ${DATA_DIR}/docs_stats.json'")

echo "docs_job=${DOCS_JOB}"
echo "${DOCS_JOB}" > "${RUN_DIR}/job_docs.txt"
