#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
RUN=/scratch/users/nzhao2/agent-runs/colmlm-1b-corpus-20260801-124745
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging/scripts/farmshare/colmlm_1b
SRC=/mnt/c/alpha_ai/edullm/scripts/farmshare/colmlm_1b

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${SRC}/fetch_entries_db.py" \
  "${SRC}/fetch_sample100bt.py" \
  "${SRC}/resubmit_entries.sh" \
  "${SRC}/watch_ingest_then_process.sh" \
  "${SRC}/submit_process.sh" \
  "${SRC}/span_extract_worker.py" \
  "${SRC}/select_docs.py" \
  "${SRC}/hash_sample.py" \
  "${SRC}/plan_span_ranges.py" \
  "${SRC}/mark_and_write.py" \
  "${SRC}/qa_report.py" \
  "${HOST}:${STAGING}/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "sed -i 's/\r\$//' ${STAGING}/*.sh ${STAGING}/*.py; chmod +x ${STAGING}/*.sh"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "RUN_DIR=${RUN} SCRIPT_DIR=${STAGING} bash ${STAGING}/resubmit_entries.sh"

WATCH=$(ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "sbatch --parsable --exclude=wheat-01 --partition=normal --cpus-per-task=1 --mem=2G --time=36:00:00 --job-name=colmlm-watch --chdir=${RUN} --output=${RUN}/logs/watch-%j.out --error=${RUN}/logs/watch-%j.err --wrap='bash -lc \"set -Eeuo pipefail; RUN_DIR=${RUN} SCRIPT_DIR=${STAGING} POLL_SECS=180 bash ${STAGING}/watch_ingest_then_process.sh\"'")

echo "watch_job=${WATCH}"
ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "echo ${WATCH} > ${RUN}/job_watch.txt; squeue -u nzhao2 -n colmlm-entries,colmlm-s100,colmlm-watch -o '%.18i %.12j %.2t %.10M %R' | head -50"
