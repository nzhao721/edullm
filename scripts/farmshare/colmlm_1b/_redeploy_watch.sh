#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
RUN=/scratch/users/nzhao2/agent-runs/colmlm-1b-corpus-20260801-124745
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging/scripts/farmshare/colmlm_1b
SRC=/mnt/c/alpha_ai/edullm/scripts/farmshare/colmlm_1b

# Cancel old watcher
OLD_WATCH=$(ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "cat ${RUN}/job_watch.txt 2>/dev/null || true")
if [[ -n "${OLD_WATCH}" ]]; then
  ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "scancel ${OLD_WATCH} || true"
  echo "cancelled old watch job ${OLD_WATCH}"
fi

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${SRC}/submit_docs.sh" \
  "${SRC}/submit_spans_and_mark.sh" \
  "${SRC}/submit_process.sh" \
  "${SRC}/watch_ingest_then_process.sh" \
  "${HOST}:${STAGING}/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "sed -i 's/\r\$//' ${STAGING}/*.sh; chmod +x ${STAGING}/*.sh"

WATCH=$(ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "sbatch --parsable --exclude=wheat-01 --partition=normal --cpus-per-task=1 --mem=2G --time=36:00:00 --job-name=colmlm-watch --chdir=${RUN} --output=${RUN}/logs/watch-%j.out --error=${RUN}/logs/watch-%j.err --wrap='bash -lc \"set -Eeuo pipefail; RUN_DIR=${RUN} SCRIPT_DIR=${STAGING} POLL_SECS=60 SPAN_WORKERS=256 SPAN_THROTTLE=128 bash ${STAGING}/watch_ingest_then_process.sh\"'")

echo "watch_job=${WATCH}"
ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "echo ${WATCH} > ${RUN}/job_watch.txt; sleep 3; bash /tmp/_status2.sh 2>/dev/null || true; squeue -u nzhao2 -n colmlm-entries,colmlm-s100,colmlm-watch,colmlm-docs -o '%.18i %.12j %.2t %.10M %R' | head -40; echo ---; n=\$(find ${RUN}/data/s100 -name '*.parquet' | wc -l); echo parquet=\$n; tail -15 ${RUN}/logs/watch-${WATCH}.out 2>/dev/null || true"
