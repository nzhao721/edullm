#!/usr/bin/env bash
# Submit document-level RegMix LM labeling on FarmShare.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
: "${RUN_DIR:=/scratch/users/${SUNET}/agent-runs/regmix-10b-20260725-124810}"
: "${VENV:=/scratch/users/${SUNET}/agent-runs/olmo-ladder-370m-20260722-185217/venv}"
: "${BASE_CONFIG:=/scratch/users/${SUNET}/agent-runs/olmo-ladder-370m-20260722-185217/checkpoints/edullm-370M-30B/step5000-unsharded/config.yaml}"
: "${REFHQ_ROOT:=/scratch/users/${SUNET}/agent-runs/refhq-models-all-20260727T220851Z/unsharded}"
: "${LM_ROOT:=${RUN_DIR}/lm_labels}"
: "${LM_LABELS_ROOT:=${LM_ROOT}/labels}"
: "${TARGET_TOKENS_PER_CHUNK:=25000000}"
: "${GPU_TIME_LIMIT:=00:45:00}"
: "${GPU_MEM:=80G}"
: "${BATCH_TOKENS:=4096}"
: "${MAX_IN_FLIGHT:=28}"
: "${POLL_SECS:=30}"
: "${ARRAY_CONCURRENCY:=4}"
: "${USE_CONTROLLER:=1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLMO_DIR=""
if [[ -d "${SCRIPT_DIR}/../olmo" ]]; then
  OLMO_DIR="$(cd "${SCRIPT_DIR}/../olmo" && pwd)"
fi

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${LM_ROOT}" "${LM_LABELS_ROOT}"

copy_script() {
  local name=$1 src
  src="${SCRIPT_DIR}/${name}"
  if [[ ! -f "${src}" && -n "${OLMO_DIR}" ]]; then
    src="${OLMO_DIR}/${name}"
  fi
  if [[ ! -f "${src}" ]]; then
    echo "missing script: ${name}" >&2
    exit 1
  fi
  cp -f "${src}" "${RUN_DIR}/scripts/${name}"
}

for name in \
  build_regmix_label_manifest.py \
  build_regmix_lm_chunks.py \
  average_refhq_checkpoints.py \
  label_regmix_doc_lm.py \
  label_regmix_doc_lm_chunk.sbatch \
  build_regmix_lm_retry_indices.py \
  finalize_regmix_lm_labels.py \
  prepare_regmix_lm_labels.sbatch \
  label_regmix_doc_lm.sbatch \
  control_regmix_doc_lm.sbatch; do
  copy_script "${name}"
done
chmod +x "${RUN_DIR}/scripts/"*.py "${RUN_DIR}/scripts/"*.sbatch 2>/dev/null || true

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "missing venv at ${VENV}" >&2
  exit 1
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "missing base config: ${BASE_CONFIG}" >&2
  exit 1
fi
for step in 250 1000 1125 1315; do
  if [[ ! -f "${REFHQ_ROOT}/step${step}/model_eval.pt" ]]; then
    echo "missing RefHQ model_eval.pt for step${step}: ${REFHQ_ROOT}/step${step}/model_eval.pt" >&2
    exit 1
  fi
done

source "${VENV}/bin/activate"
export PYTHONPATH="${RUN_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

SHARD_MANIFEST="${LM_ROOT}/regmix_shard_manifest.jsonl"
python "${RUN_DIR}/scripts/build_regmix_label_manifest.py" --run-dir "${RUN_DIR}" --out "${SHARD_MANIFEST}"

WORK_MANIFEST="${LM_ROOT}/lm_work_manifest.jsonl"
LATE_AVG_CHECKPOINT="${LM_ROOT}/checkpoints/refhq-step1000-1125-1315-avg.model_eval.pt"
EARLY_CHECKPOINT="${REFHQ_ROOT}/step250/model_eval.pt"

if [[ ! -f "${WORK_MANIFEST}" || ! -f "${LATE_AVG_CHECKPOINT}" || "${FORCE_PREP:-0}" == "1" ]]; then
  PREP_JOB="$(sbatch --parsable \
    --exclude=wheat-01 \
    --partition=normal \
    --cpus-per-task=8 \
    --mem=64G \
    --time="${PREP_TIME_LIMIT:-01:00:00}" \
    --chdir="${RUN_DIR}" \
    --export=ALL,RUN_DIR="${RUN_DIR}",VENV="${VENV}",SHARD_MANIFEST="${SHARD_MANIFEST}",LM_ROOT="${LM_ROOT}",REFHQ_ROOT="${REFHQ_ROOT}",TARGET_TOKENS_PER_CHUNK="${TARGET_TOKENS_PER_CHUNK}" \
    "${RUN_DIR}/scripts/prepare_regmix_lm_labels.sbatch")"
  cat >"${LM_ROOT}/prep_job.json" <<EOF
{
  "prep_job": "${PREP_JOB}",
  "run_dir": "${RUN_DIR}",
  "lm_root": "${LM_ROOT}",
  "work_manifest": "${WORK_MANIFEST}",
  "late_avg_checkpoint": "${LATE_AVG_CHECKPOINT}",
  "target_tokens_per_chunk": ${TARGET_TOKENS_PER_CHUNK}
}
EOF
  printf 'prep_job=%s\n' "${PREP_JOB}"
  printf 'prep_artifacts=%s and %s\n' "${WORK_MANIFEST}" "${LATE_AVG_CHECKPOINT}"
  printf 'rerun this script after prep succeeds to submit the GPU label array\n'
  exit 0
fi

N="$(wc -l < "${WORK_MANIFEST}" | tr -d ' ')"
if [[ "${N}" -lt 1 ]]; then
  echo "empty work manifest: ${WORK_MANIFEST}" >&2
  exit 1
fi

# FarmShare gpu QOS: MaxSubmitPU=32, MaxJobsPU=4. Use a pipeline controller that
# keeps up to MAX_IN_FLIGHT chunk jobs submitted and refills as slots open.
if [[ "${USE_CONTROLLER}" == "1" ]]; then
  CTRL_JOB="$(sbatch --parsable \
    --exclude=wheat-01 \
    --partition=normal \
    --cpus-per-task=2 \
    --mem=4G \
    --time=48:00:00 \
    --chdir="${RUN_DIR}" \
    --export=ALL,RUN_DIR="${RUN_DIR}",VENV="${VENV}",WORK_MANIFEST="${WORK_MANIFEST}",LM_LABELS_ROOT="${LM_LABELS_ROOT}",EARLY_CHECKPOINT="${EARLY_CHECKPOINT}",LATE_AVG_CHECKPOINT="${LATE_AVG_CHECKPOINT}",BASE_CONFIG="${BASE_CONFIG}",BATCH_TOKENS="${BATCH_TOKENS}",LM_ROOT="${LM_ROOT}",MAX_IN_FLIGHT="${MAX_IN_FLIGHT}",GPU_TIME_LIMIT="${GPU_TIME_LIMIT}",GPU_MEM="${GPU_MEM}",POLL_SECS="${POLL_SECS}",RUN_FINALIZE="${RUN_FINALIZE:-1}" \
    "${RUN_DIR}/scripts/control_regmix_doc_lm.sbatch")"
  cat >"${LM_ROOT}/jobs.json" <<EOF
{
  "pipeline_controller_job": "${CTRL_JOB}",
  "n_chunks": ${N},
  "max_in_flight": ${MAX_IN_FLIGHT},
  "poll_secs": ${POLL_SECS},
  "gpu_time_limit": "${GPU_TIME_LIMIT}",
  "gpu_mem": "${GPU_MEM}",
  "labels_root": "${LM_LABELS_ROOT}",
  "run_dir": "${RUN_DIR}",
  "work_manifest": "${WORK_MANIFEST}",
  "early_checkpoint": "${EARLY_CHECKPOINT}",
  "late_avg_checkpoint": "${LATE_AVG_CHECKPOINT}",
  "wheat_01_excluded": true
}
EOF
  printf 'pipeline_controller_job=%s (%s chunks, max_in_flight=%s)\n' \
    "${CTRL_JOB}" "${N}" "${MAX_IN_FLIGHT}"
  printf 'labels_root=%s\n' "${LM_LABELS_ROOT}"
  exit 0
fi

ARRAY_MAX=$((N - 1))
if [[ "${ARRAY_CONCURRENCY}" -gt "${N}" ]]; then
  ARRAY_CONCURRENCY="${N}"
fi

LABEL_JOB="$(sbatch --parsable \
  --exclude=wheat-01 \
  --partition=gpu \
  --qos=gpu \
  --gpus-per-node=1 \
  --cpus-per-task=8 \
  --mem="${GPU_MEM}" \
  --time="${GPU_TIME_LIMIT}" \
  --array="0-${ARRAY_MAX}%${ARRAY_CONCURRENCY}" \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",VENV="${VENV}",WORK_MANIFEST="${WORK_MANIFEST}",LM_LABELS_ROOT="${LM_LABELS_ROOT}",EARLY_CHECKPOINT="${EARLY_CHECKPOINT}",LATE_AVG_CHECKPOINT="${LATE_AVG_CHECKPOINT}",BASE_CONFIG="${BASE_CONFIG}",BATCH_TOKENS="${BATCH_TOKENS}" \
  "${RUN_DIR}/scripts/label_regmix_doc_lm.sbatch")"

FINAL_JOB="$(sbatch --parsable \
  --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task=4 \
  --mem=16G \
  --time=02:00:00 \
  --job-name=regmix-lm-final \
  --dependency="afterok:${LABEL_JOB}" \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/lm-finalize-%j.out" \
  --error="${RUN_DIR}/logs/lm-finalize-%j.err" \
  --wrap="source '${VENV}/bin/activate'; export PYTHONPATH='${RUN_DIR}/scripts'; python '${RUN_DIR}/scripts/finalize_regmix_lm_labels.py' --labels-root '${LM_LABELS_ROOT}' --work-manifest '${WORK_MANIFEST}'")"

cat >"${LM_ROOT}/jobs.json" <<EOF
{
  "label_array_job": "${LABEL_JOB}",
  "finalize_job": "${FINAL_JOB}",
  "n_chunks": ${N},
  "array_concurrency": ${ARRAY_CONCURRENCY},
  "gpu_time_limit": "${GPU_TIME_LIMIT}",
  "gpu_mem": "${GPU_MEM}",
  "labels_root": "${LM_LABELS_ROOT}",
  "run_dir": "${RUN_DIR}",
  "work_manifest": "${WORK_MANIFEST}",
  "early_checkpoint": "${EARLY_CHECKPOINT}",
  "late_avg_checkpoint": "${LATE_AVG_CHECKPOINT}",
  "wheat_01_excluded": true
}
EOF

printf 'label_array_job=%s (%s chunks, concurrency=%s)\n' "${LABEL_JOB}" "${N}" "${ARRAY_CONCURRENCY}"
printf 'finalize_job=%s\n' "${FINAL_JOB}"
printf 'labels_root=%s\n' "${LM_LABELS_ROOT}"
