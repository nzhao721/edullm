#!/usr/bin/env bash
# Watch ingest: submit Pass B as soon as s100 is ready; Pass A+mark when entries.db ready.
set -Eeuo pipefail
SUNET="${SUNET:-nzhao2}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
SCRIPT_DIR="${SCRIPT_DIR:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging/scripts/farmshare/colmlm_1b}"
POLL_SECS="${POLL_SECS:-120}"
SPAN_WORKERS="${SPAN_WORKERS:-256}"
SPAN_THROTTLE="${SPAN_THROTTLE:-128}"

entries_id=$(cat "${RUN_DIR}/job_entries.txt")
s100_id=$(cat "${RUN_DIR}/job_s100.txt")

echo "watching entries=${entries_id} s100=${s100_id}"
echo "SPAN_WORKERS=${SPAN_WORKERS} SPAN_THROTTLE=${SPAN_THROTTLE}"

wait_job() {
  local jid="$1"
  while true; do
    local state
    state=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null | sort -u | tr '\n' ',' || true)
    if [[ -z "${state}" ]]; then
      local unfinished
      unfinished=$(sacct -j "${jid}" --state=PENDING,RUNNING,COMPLETING -n 2>/dev/null | wc -l || true)
      if [[ "${unfinished}" -gt 0 ]]; then
        sleep "${POLL_SECS}"
        continue
      fi
      # Count only array/batch leaf tasks (JobID contains '_' or has no '.') that failed.
      local failed
      failed=$(sacct -j "${jid}" -n -o JobID,State -P 2>/dev/null \
        | awk -F'|' '$1 ~ /_/ && $1 !~ /\./ && $2 ~ /FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL/ {c++} END{print c+0}' \
        || true)
      if [[ "${failed}" -gt 0 ]]; then
        echo "ERROR: job ${jid} had ${failed} failed leaf tasks" >&2
        sacct -j "${jid}" -o JobID,State,ExitCode,Elapsed -P 2>/dev/null | head -80 >&2 || true
        return 1
      fi
      echo "job ${jid} completed ok"
      return 0
    fi
    echo "$(date -u +%FT%TZ) ${jid} still in queue states=${state}"
    sleep "${POLL_SECS}"
  done
}

# --- Pass B early: as soon as sample-100BT is ready ---
wait_job "${s100_id}"
n_parq=$(find "${RUN_DIR}/data/s100" -name '*.parquet' | wc -l)
echo "s100 parquet count=${n_parq}"
if [[ "${n_parq}" -lt 140 ]]; then
  echo "ERROR: expected 140 parquet, got ${n_parq}" >&2
  exit 2
fi
echo "submitting Pass B (docs) early"
RUN_DIR="${RUN_DIR}" SCRIPT_DIR="${SCRIPT_DIR}" bash "${SCRIPT_DIR}/submit_docs.sh"

# --- Pass A + mark: after entries.db ---
wait_job "${entries_id}"
db="${RUN_DIR}/data/fineweb_with_fullwiki_entries.db"
sz=$(stat -c%s "${db}")
echo "entries.db size=${sz}"
if [[ "${sz}" -lt 700000000000 ]]; then
  echo "ERROR: entries.db too small" >&2
  exit 2
fi
echo "submitting Pass A (spans x${SPAN_WORKERS}) + mark"
RUN_DIR="${RUN_DIR}" SCRIPT_DIR="${SCRIPT_DIR}" \
  SPAN_WORKERS="${SPAN_WORKERS}" SPAN_THROTTLE="${SPAN_THROTTLE}" \
  bash "${SCRIPT_DIR}/submit_spans_and_mark.sh"

echo "all process stages submitted"
