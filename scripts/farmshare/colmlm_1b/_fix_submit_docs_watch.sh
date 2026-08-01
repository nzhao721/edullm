#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
RUN=/scratch/users/nzhao2/agent-runs/colmlm-1b-corpus-20260801-124745
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging/scripts/farmshare/colmlm_1b
SRC=/mnt/c/alpha_ai/edullm/scripts/farmshare/colmlm_1b

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${SRC}/submit_docs.sh" \
  "${SRC}/submit_spans_and_mark.sh" \
  "${SRC}/watch_ingest_then_process.sh" \
  "${HOST}:${STAGING}/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "sed -i 's/\r\$//' ${STAGING}/*.sh; chmod +x ${STAGING}/*.sh"

# Submit Pass B immediately (s100 is ready)
ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "RUN_DIR=${RUN} SCRIPT_DIR=${STAGING} bash ${STAGING}/submit_docs.sh"

# Restart watcher for entries → spans×256 + mark
OLD=$(ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "cat ${RUN}/job_watch.txt 2>/dev/null || true")
[[ -n "${OLD}" ]] && ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "scancel ${OLD} 2>/dev/null || true"

WATCH=$(ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "sbatch --parsable --exclude=wheat-01 --partition=normal --cpus-per-task=1 --mem=2G --time=36:00:00 --job-name=colmlm-watch --chdir=${RUN} --output=${RUN}/logs/watch-%j.out --error=${RUN}/logs/watch-%j.err --wrap='bash -lc \"set -Eeuo pipefail; RUN_DIR=${RUN} SCRIPT_DIR=${STAGING} POLL_SECS=120 SPAN_WORKERS=256 SPAN_THROTTLE=128 bash ${STAGING}/watch_ingest_then_process.sh\"'")

echo "watch_job=${WATCH}"
ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "echo ${WATCH} > ${RUN}/job_watch.txt; echo docs=\$(cat ${RUN}/job_docs.txt); squeue -u nzhao2 -o '%.18i %.12j %.2t %.10M %R' | grep -E 'JOBID|colmlm' || true"
