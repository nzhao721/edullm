#!/usr/bin/env bash
# Submit pipeline retry for missing RegMix document-level LM label chunks.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
: "${RUN_DIR:=/scratch/users/${SUNET}/agent-runs/regmix-10b-20260725-124810}"
: "${VENV:=/scratch/users/${SUNET}/agent-runs/olmo-ladder-370m-20260722-185217/venv}"
: "${BASE_CONFIG:=/scratch/users/${SUNET}/agent-runs/olmo-ladder-370m-20260722-185217/checkpoints/edullm-370M-30B/step5000-unsharded/config.yaml}"
: "${REFHQ_ROOT:=/scratch/users/${SUNET}/agent-runs/refhq-models-all-20260727T220851Z/unsharded}"
: "${LM_ROOT:=${RUN_DIR}/lm_labels}"
: "${LM_LABELS_ROOT:=${LM_ROOT}/labels}"
: "${MAX_IN_FLIGHT:=28}"
: "${ARRAY_BATCH_SIZE:=28}"
: "${ARRAY_CONCURRENCY:=4}"
: "${GPU_TIME_LIMIT:=00:45:00}"
: "${GPU_MEM:=64G}"
: "${GPU_CPUS_PER_TASK:=4}"
: "${BATCH_TOKENS:=2048}"
: "${POLL_SECS:=30}"
: "${RUN_FINALIZE:=0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${LM_ROOT}" "${LM_LABELS_ROOT}"

if [[ "$(cd "${SCRIPT_DIR}" && pwd -P)" != "$(cd "${RUN_DIR}/scripts" && pwd -P)" ]]; then
  for name in \
    build_regmix_lm_retry_indices.py \
    label_regmix_doc_lm.py \
    label_regmix_doc_lm_retry.sbatch \
    control_regmix_doc_lm_retry.sbatch \
    finalize_regmix_lm_labels.py; do
    cp -f "${SCRIPT_DIR}/${name}" "${RUN_DIR}/scripts/${name}"
  done
fi
chmod +x "${RUN_DIR}/scripts/"*.py "${RUN_DIR}/scripts/"*.sbatch 2>/dev/null || true

source "${VENV}/bin/activate"
export PYTHONPATH="${RUN_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

WORK_MANIFEST="${LM_ROOT}/lm_work_manifest.jsonl"
LATE_AVG_CHECKPOINT="${LM_ROOT}/checkpoints/refhq-step1000-1125-1315-avg.model_eval.pt"
EARLY_CHECKPOINT="${REFHQ_ROOT}/step250/model_eval.pt"
RETRY_INDICES="${LM_ROOT}/retry_indices.txt"
FAILED_INDICES="${LM_ROOT}/failed_indices.txt"
touch "${FAILED_INDICES}"

python "${RUN_DIR}/scripts/build_regmix_lm_retry_indices.py" \
  --work-manifest "${WORK_MANIFEST}" \
  --labels-root "${LM_LABELS_ROOT}" \
  --failed-indices "${FAILED_INDICES}" \
  --out "${RETRY_INDICES}"
N_MISSING="$(wc -l < "${RETRY_INDICES}" | tr -d ' ')"
if [[ "${N_MISSING}" -lt 1 ]]; then
  echo "no missing chunks to retry"
  exit 0
fi

RETRY_JOB="$(sbatch --parsable \
  --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task=2 \
  --mem=4G \
  --time=48:00:00 \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",VENV="${VENV}",WORK_MANIFEST="${WORK_MANIFEST}",LM_LABELS_ROOT="${LM_LABELS_ROOT}",EARLY_CHECKPOINT="${EARLY_CHECKPOINT}",LATE_AVG_CHECKPOINT="${LATE_AVG_CHECKPOINT}",BASE_CONFIG="${BASE_CONFIG}",BATCH_TOKENS="${BATCH_TOKENS}",LM_ROOT="${LM_ROOT}",MAX_IN_FLIGHT="${MAX_IN_FLIGHT}",ARRAY_BATCH_SIZE="${ARRAY_BATCH_SIZE}",ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY}",GPU_CPUS_PER_TASK="${GPU_CPUS_PER_TASK:-4}",GPU_TIME_LIMIT="${GPU_TIME_LIMIT}",GPU_MEM="${GPU_MEM}",POLL_SECS="${POLL_SECS}",RUN_FINALIZE="${RUN_FINALIZE}" \
  "${RUN_DIR}/scripts/control_regmix_doc_lm_retry.sbatch")"

cat >"${LM_ROOT}/retry_submit.json" <<EOF
{
  "pipeline_controller_job": "${RETRY_JOB}",
  "n_missing": ${N_MISSING},
  "retry_indices": "${RETRY_INDICES}",
  "max_in_flight": ${MAX_IN_FLIGHT},
  "poll_secs": ${POLL_SECS}
}
EOF

printf 'pipeline_controller_job=%s (n_missing=%s max_in_flight=%s)\n' \
  "${RETRY_JOB}" "${N_MISSING}" "${MAX_IN_FLIGHT}"
printf 'retry_indices=%s\n' "${RETRY_INDICES}"
