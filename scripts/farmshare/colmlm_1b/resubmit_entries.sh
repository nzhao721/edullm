#!/usr/bin/env bash
# Resubmit only the entries.db download (keeps any .partial resume bytes).
set -Eeuo pipefail
SUNET="${SUNET:-nzhao2}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
SCRIPT_DIR="${SCRIPT_DIR:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging/scripts/farmshare/colmlm_1b}"
VENV="${VENV:-/scratch/users/${SUNET}/agent-runs/venvs/colmlm-1b}"
EXCLUDE="${EXCLUDE:-wheat-01}"

# Cancel previous entries job if still running
if [[ -f "${RUN_DIR}/job_entries.txt" ]]; then
  old=$(cat "${RUN_DIR}/job_entries.txt")
  scancel "${old}" 2>/dev/null || true
fi

ENTRIES_JOB=$(sbatch --parsable --exclude="${EXCLUDE}" \
  --partition=normal \
  --cpus-per-task=4 \
  --mem=16G \
  --time=24:00:00 \
  --job-name=colmlm-entries \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/entries-%j.out" \
  --error="${RUN_DIR}/logs/entries-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; source ${RUN_DIR}/env.sh; source ${VENV}/bin/activate; python -u ${SCRIPT_DIR}/fetch_entries_db.py --out ${RUN_DIR}/data/fineweb_with_fullwiki_entries.db --chunk-mb 128'")

echo "entries_job=${ENTRIES_JOB}"
echo "${ENTRIES_JOB}" > "${RUN_DIR}/job_entries.txt"
