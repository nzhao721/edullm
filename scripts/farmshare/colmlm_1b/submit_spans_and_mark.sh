#!/usr/bin/env bash
# Pass A (span extract, high parallel) + mark/QA (depends on docs + spans).
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV="${VENV:-/scratch/users/${SUNET}/agent-runs/venvs/colmlm-1b}"
DATA_DIR="${DATA_DIR:-${RUN_DIR}/data}"
ENTRIES_DB="${ENTRIES_DB:-${DATA_DIR}/fineweb_with_fullwiki_entries.db}"
SPANS_DIR="${SPANS_DIR:-${DATA_DIR}/spans}"
DOCS_PARQUET="${DOCS_PARQUET:-${DATA_DIR}/docs_sample.parquet}"
CORPUS_DIR="${CORPUS_DIR:-${RUN_DIR}/corpus_1b}"
SPAN_WORKERS="${SPAN_WORKERS:-256}"
SPAN_THROTTLE="${SPAN_THROTTLE:-128}"
EXCLUDE="${EXCLUDE:-wheat-01}"
MODULUS="${MODULUS:-100}"
RESIDUE="${RESIDUE:-0}"
MARK_CPUS="${MARK_CPUS:-48}"
MARK_MEM="${MARK_MEM:-160G}"

mkdir -p "${RUN_DIR}/logs" "${SPANS_DIR}" "${CORPUS_DIR}"

# shellcheck disable=SC1091
source "${RUN_DIR}/env.sh"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [[ ! -f "${ENTRIES_DB}" ]]; then
  echo "ERROR: missing ${ENTRIES_DB}" >&2
  exit 2
fi
sz=$(stat -c%s "${ENTRIES_DB}")
if [[ "${sz}" -lt 700000000000 ]]; then
  echo "ERROR: entries.db too small (${sz})" >&2
  exit 2
fi

# Resolve docs job dependency (must exist — submitted earlier by watcher/submit_docs)
if [[ ! -f "${RUN_DIR}/job_docs.txt" ]]; then
  echo "ERROR: job_docs.txt missing; submit_docs.sh first" >&2
  exit 2
fi
DOCS_JOB=$(cat "${RUN_DIR}/job_docs.txt")

python -u "${SCRIPT_DIR}/plan_span_ranges.py" \
  --db "${ENTRIES_DB}" \
  --workers "${SPAN_WORKERS}" \
  --out "${DATA_DIR}/span_ranges.json"

N_RANGES=$(python - <<PY
import json
print(json.load(open("${DATA_DIR}/span_ranges.json"))["workers"])
PY
)

SPANS_JOB=$(sbatch --parsable --exclude="${EXCLUDE}" \
  --partition=normal \
  --array=0-$((N_RANGES - 1))%"${SPAN_THROTTLE}" \
  --cpus-per-task=2 \
  --mem=8G \
  --time=4:00:00 \
  --job-name=colmlm-spans \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/spans-%A_%a.out" \
  --error="${RUN_DIR}/logs/spans-%A_%a.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; source ${RUN_DIR}/env.sh; source ${VENV}/bin/activate; python - <<PY
import json, os, subprocess, sys
from pathlib import Path
ranges = json.load(open(\"${DATA_DIR}/span_ranges.json\"))[\"ranges\"]
i = int(os.environ[\"SLURM_ARRAY_TASK_ID\"])
r = ranges[i]
out = Path(\"${SPANS_DIR}\") / f\"spans_{i:04d}.parquet\"
if out.is_file() and out.stat().st_size > 0:
    print(\"skip existing\", out, flush=True)
    raise SystemExit(0)
cmd = [
  sys.executable, \"${SCRIPT_DIR}/span_extract_worker.py\",
  \"--db\", \"${ENTRIES_DB}\",
  \"--lo\", str(r[\"lo\"]), \"--hi\", str(r[\"hi\"]),
  \"--out\", str(out),
  \"--modulus\", \"${MODULUS}\", \"--residue\", \"${RESIDUE}\",
]
raise SystemExit(subprocess.call(cmd))
PY
'")

# Mark waits for docs job AND spans array
MARK_JOB=$(sbatch --parsable --exclude="${EXCLUDE}" \
  --partition=normal \
  --cpus-per-task="${MARK_CPUS}" \
  --mem="${MARK_MEM}" \
  --time=3:00:00 \
  --job-name=colmlm-mark \
  --dependency=afterok:${DOCS_JOB}:${SPANS_JOB} \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/mark-%j.out" \
  --error="${RUN_DIR}/logs/mark-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; source ${RUN_DIR}/env.sh; source ${VENV}/bin/activate; python -u ${SCRIPT_DIR}/mark_and_write.py --docs ${DOCS_PARQUET} --spans-glob \"${SPANS_DIR}/spans_*.parquet\" --out-dir ${CORPUS_DIR} --workers ${MARK_CPUS} --stats-out ${DATA_DIR}/mark_stats.json; python -u ${SCRIPT_DIR}/qa_report.py --docs-stats ${DATA_DIR}/docs_stats.json --mark-stats ${DATA_DIR}/mark_stats.json --corpus-glob \"${CORPUS_DIR}/dump=*/part-*.parquet\" --out ${RUN_DIR}/qa_report.json'")

echo "spans_job=${SPANS_JOB} workers=${N_RANGES} throttle=${SPAN_THROTTLE}"
echo "mark_job=${MARK_JOB} (afterok docs=${DOCS_JOB} spans=${SPANS_JOB})"
echo "${SPANS_JOB}" > "${RUN_DIR}/job_spans.txt"
echo "${MARK_JOB}" > "${RUN_DIR}/job_mark.txt"
