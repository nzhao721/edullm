#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=regmix-lm-pipe
#SBATCH --output=logs/lm-pipe-ctrl-%j.out
#SBATCH --error=logs/lm-pipe-ctrl-%j.err

# Overlapping GPU array batches with %4 concurrency (Slurm refills within each array).
# Unlike the old wave controller, do NOT wait for an entire batch to finish before
# submitting the next disjoint batch. Unlike single-chunk sbatch, arrays get correct
# per-task GPU binding on shared nodes.
set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${WORK_MANIFEST:?}" "${LM_LABELS_ROOT:?}"
: "${EARLY_CHECKPOINT:?}" "${LATE_AVG_CHECKPOINT:?}" "${BASE_CONFIG:?}" "${LM_ROOT:?}"

MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-28}"
ARRAY_BATCH_SIZE="${ARRAY_BATCH_SIZE:-28}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-4}"
GPU_TIME_LIMIT="${GPU_TIME_LIMIT:-00:45:00}"
GPU_MEM="${GPU_MEM:-80G}"
BATCH_TOKENS="${BATCH_TOKENS:-8192}"
POLL_SECS="${POLL_SECS:-30}"
RUN_FINALIZE="${RUN_FINALIZE:-0}"
GPU_JOB_NAME="${GPU_JOB_NAME:-regmix-lm-retry}"

cd "${RUN_DIR}"
mkdir -p "${RUN_DIR}/logs" "${LM_LABELS_ROOT}" "${LM_ROOT}/active_batches"
source "${VENV}/bin/activate"
export PYTHONPATH="${RUN_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

RETRY_INDICES="${LM_ROOT}/retry_indices.txt"
FAILED_INDICES="${LM_ROOT}/failed_indices.txt"
ACTIVE_DIR="${LM_ROOT}/active_batches"
touch "${FAILED_INDICES}"

record_array_failures() {
  local job_id="$1"
  local snapshot="$2"
  local -a chunks=()
  local line task_id chunk_idx state
  [[ -f "${snapshot}" ]] || return 0
  while IFS= read -r line; do
    [[ -n "${line}" ]] && chunks+=("${line}")
  done <"${snapshot}"
  while IFS='|' read -r task state _rest; do
    [[ "${task}" =~ ^${job_id}_([0-9]+)$ ]] || continue
    task_id="${BASH_REMATCH[1]}"
    case "${state}" in
      COMPLETED) continue ;;
    esac
    chunk_idx="${chunks[${task_id}]:-}"
    [[ -n "${chunk_idx}" ]] || continue
    if ! grep -qx "${chunk_idx}" "${FAILED_INDICES}" 2>/dev/null; then
      printf '%s\n' "${chunk_idx}" >>"${FAILED_INDICES}"
      echo "record failed chunk=${chunk_idx} task=${task} state=${state}"
    fi
  done < <(sacct -j "${job_id}" --format=JobID,State -P -n 2>/dev/null)
}

prune_arrays() {
  local f job_id snapshot
  for f in "${ACTIVE_DIR}"/*.job; do
    [[ -e "${f}" ]] || continue
    job_id="$(basename "${f}" .job)"
    if squeue -j "${job_id}" -h 2>/dev/null | grep -q .; then
      continue
    fi
    snapshot="$(cat "${f}")"
    record_array_failures "${job_id}" "${snapshot}"
    echo "prune array job=${job_id}"
    rm -f "${f}" "${snapshot}"
  done
  rm -f "${ACTIVE_DIR}/.job" "${ACTIVE_DIR}/.indices" 2>/dev/null || true
}

chunk_is_active() {
  local chunk_index="$1"
  local g
  for g in "${ACTIVE_DIR}"/*.indices; do
    [[ -e "${g}" ]] || continue
    if grep -qx "${chunk_index}" "${g}" 2>/dev/null; then
      return 0
    fi
  done
  for g in "${ACTIVE_DIR}"/*.job; do
    [[ -e "${g}" ]] || continue
    snapshot="$(cat "${g}")"
    if [[ -f "${snapshot}" ]] && grep -qx "${chunk_index}" "${snapshot}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

build_batch_snapshot() {
  local out="$1"
  local limit="$2"
  local chunk_index n=0
  : >"${out}"
  while IFS= read -r chunk_index; do
    [[ -n "${chunk_index}" ]] || continue
    if chunk_is_active "${chunk_index}"; then
      continue
    fi
    printf '%s\n' "${chunk_index}" >>"${out}"
    n=$((n + 1))
    if [[ "${n}" -ge "${limit}" ]]; then
      break
    fi
  done <"${RETRY_INDICES}"
  printf '%s' "${n}"
}

submit_array_batch() {
  local snapshot n_tasks max_index job_id token
  token="$(date +%s)-$$-${RANDOM}"
  snapshot="${ACTIVE_DIR}/batch-${token}.indices"
  n_tasks="$(build_batch_snapshot "${snapshot}" "${ARRAY_BATCH_SIZE}")"
  if [[ "${n_tasks}" -lt 1 ]]; then
    rm -f "${snapshot}"
    return 1
  fi
  max_index=$((n_tasks - 1))
  job_id="$(sbatch --parsable \
    --exclude=wheat-01 \
    --partition=gpu \
    --qos=gpu \
    --gpus-per-node=1 \
    --cpus-per-task=8 \
    --mem="${GPU_MEM}" \
    --time="${GPU_TIME_LIMIT}" \
    --array="0-${max_index}%${ARRAY_CONCURRENCY}" \
    --job-name="${GPU_JOB_NAME}" \
    --chdir="${RUN_DIR}" \
    --export=ALL,RUN_DIR="${RUN_DIR}",VENV="${VENV}",WORK_MANIFEST="${WORK_MANIFEST}",LM_LABELS_ROOT="${LM_LABELS_ROOT}",EARLY_CHECKPOINT="${EARLY_CHECKPOINT}",LATE_AVG_CHECKPOINT="${LATE_AVG_CHECKPOINT}",BASE_CONFIG="${BASE_CONFIG}",BATCH_TOKENS="${BATCH_TOKENS}",RETRY_INDICES="${snapshot}" \
    "${RUN_DIR}/scripts/label_regmix_doc_lm_retry.sbatch")"
  if [[ -z "${job_id}" ]]; then
    echo "sbatch failed for snapshot=${snapshot}" >&2
    rm -f "${snapshot}"
    return 1
  fi
  printf '%s\n' "${snapshot}" >"${ACTIVE_DIR}/${job_id}.job"
  echo "submit array job=${job_id} tasks=${n_tasks} concurrency=${ARRAY_CONCURRENCY} snapshot=${snapshot}"
  return 0
}

gpu_tasks_queued() {
  squeue -u "${USER}" -h -n "${GPU_JOB_NAME}" -t RUNNING,PENDING 2>/dev/null | wc -l | tr -d ' '
}

active_array_count() {
  find "${ACTIVE_DIR}" -maxdepth 1 -name '*.job' 2>/dev/null | wc -l | tr -d ' '
}

echo "pipeline(array) start max_in_flight=${MAX_IN_FLIGHT} batch_size=${ARRAY_BATCH_SIZE} concurrency=${ARRAY_CONCURRENCY}"

while true; do
  python "${RUN_DIR}/scripts/build_regmix_lm_retry_indices.py" \
    --work-manifest "${WORK_MANIFEST}" \
    --labels-root "${LM_LABELS_ROOT}" \
    --failed-indices "${FAILED_INDICES}" \
    --out "${RETRY_INDICES}"
  n_missing="$(wc -l < "${RETRY_INDICES}" | tr -d ' ')"
  n_failed="$(wc -l < "${FAILED_INDICES}" | tr -d ' ')"
  prune_arrays
  queued="$(gpu_tasks_queued)"
  n_active_arrays="$(active_array_count)"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) missing=${n_missing} failed_skipped=${n_failed} gpu_queued=${queued} active_arrays=${n_active_arrays}"

  if [[ "${n_missing}" -eq 0 && "${queued}" -eq 0 && "${n_active_arrays}" -eq 0 ]]; then
    echo "pipeline complete"
    break
  fi

  submitted=0
  while [[ "${n_missing}" -gt 0 && "${queued}" -lt "${MAX_IN_FLIGHT}" ]]; do
    if ! submit_array_batch; then
      break
    fi
    submitted=$((submitted + 1))
    queued="$(gpu_tasks_queued)"
    n_missing="$(wc -l < "${RETRY_INDICES}" | tr -d ' ')"
    if [[ "${queued}" -ge "${MAX_IN_FLIGHT}" ]]; then
      break
    fi
  done

  sleep "${POLL_SECS}"
done

finalize_rc=0
if [[ "${RUN_FINALIZE}" == "1" ]]; then
  python "${RUN_DIR}/scripts/finalize_regmix_lm_labels.py" \
    --labels-root "${LM_LABELS_ROOT}" \
    --work-manifest "${WORK_MANIFEST}"
  finalize_rc=$?
fi

n_missing="$(wc -l < "${RETRY_INDICES}" | tr -d ' ')"
n_failed="$(wc -l < "${FAILED_INDICES}" | tr -d ' ')"
cat >"${LM_ROOT}/retry_jobs.json" <<EOF
{
  "pipeline_controller_job": "${SLURM_JOB_ID}",
  "n_missing_remaining": ${n_missing},
  "n_failed_skipped": ${n_failed},
  "failed_indices": "${FAILED_INDICES}",
  "max_in_flight": ${MAX_IN_FLIGHT},
  "array_batch_size": ${ARRAY_BATCH_SIZE},
  "array_concurrency": ${ARRAY_CONCURRENCY},
  "finalize_rc": ${finalize_rc},
  "run_finalize": ${RUN_FINALIZE}
}
EOF
echo "pipeline done n_missing_remaining=${n_missing} finalize_rc=${finalize_rc}"
exit 0
