#!/usr/bin/env bash
# Submit max-parallel multi-node labeling for the ~30B OLMo-mix sample.
set -Eeuo pipefail

: "${RUN_DIR:=/scratch/users/${USER}/agent-runs/olmo-mix-30b-20260722}"
: "${VENV:=${RUN_DIR}/venv}"
: "${LABELS_ROOT:=${RUN_DIR}/labels}"
: "${ARRAY_CONCURRENCY:=64}"
: "${CPUS_PER_TASK:=8}"
: "${MEM:=24G}"
: "${TIME_LIMIT:=12:00:00}"

# 64 jobs * 8 CPUs = 512 CPUs, matching QoS normal maxtrespu=cpu=512.
# Array tasks spill across all available normal nodes (wheat-01 excluded).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${LABELS_ROOT}"

# Prefer scripts already on the run dir; fall back to this checkout.
if [[ "${SCRIPT_DIR}" != "${RUN_DIR}/scripts" ]]; then
  for name in text_difficulty_metrics.py build_label_manifest.py label_olmo_shard.py \
    finalize_olmo_labels.py materialize_curriculum.py label_olmo_shard.sbatch; do
    if [[ -f "${SCRIPT_DIR}/${name}" ]]; then
      cp -f "${SCRIPT_DIR}/${name}" "${RUN_DIR}/scripts/${name}"
    fi
  done
fi
chmod +x "${RUN_DIR}/scripts/"*.py "${RUN_DIR}/scripts/"*.sbatch 2>/dev/null || true

source "${VENV}/bin/activate"
export PYTHONPATH="${RUN_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

MANIFEST="${LABELS_ROOT}/label_manifest.jsonl"
python "${RUN_DIR}/scripts/build_label_manifest.py" --run-dir "${RUN_DIR}" --out "${MANIFEST}"
N="$(wc -l < "${MANIFEST}" | tr -d ' ')"
if [[ "${N}" -lt 1 ]]; then
  echo "empty label manifest" >&2
  exit 1
fi

ARRAY_MAX=$((N - 1))
# Cap concurrency so concurrent CPUs stay within QoS 512.
MAX_BY_CPU=$((512 / CPUS_PER_TASK))
if [[ "${ARRAY_CONCURRENCY}" -gt "${MAX_BY_CPU}" ]]; then
  ARRAY_CONCURRENCY="${MAX_BY_CPU}"
fi
if [[ "${ARRAY_CONCURRENCY}" -gt 128 ]]; then
  ARRAY_CONCURRENCY=128
fi

LABEL_JOB="$(sbatch --parsable \
  --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEM}" \
  --time="${TIME_LIMIT}" \
  --array="0-${ARRAY_MAX}%${ARRAY_CONCURRENCY}" \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",VENV="${VENV}",MANIFEST="${MANIFEST}",LABELS_ROOT="${LABELS_ROOT}" \
  "${RUN_DIR}/scripts/label_olmo_shard.sbatch")"

FINAL_JOB="$(sbatch --parsable \
  --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task=4 \
  --mem=16G \
  --time=06:00:00 \
  --job-name=olmo-label-final \
  --dependency="afterok:${LABEL_JOB}" \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/label-finalize-%j.out" \
  --error="${RUN_DIR}/logs/label-finalize-%j.err" \
  --wrap="source '${VENV}/bin/activate'; export PYTHONPATH='${RUN_DIR}/scripts'; python '${RUN_DIR}/scripts/finalize_olmo_labels.py' --labels-root '${LABELS_ROOT}' --manifest '${MANIFEST}'")"

cat >"${LABELS_ROOT}/jobs.json" <<EOF
{
  "label_array_job": "${LABEL_JOB}",
  "finalize_job": "${FINAL_JOB}",
  "n_shards": ${N},
  "array_concurrency": ${ARRAY_CONCURRENCY},
  "cpus_per_task": ${CPUS_PER_TASK},
  "labels_root": "${LABELS_ROOT}",
  "wheat_01_excluded": true
}
EOF

printf 'label_array_job=%s (%s tasks, concurrency=%s, cpus/task=%s)\n' \
  "${LABEL_JOB}" "${N}" "${ARRAY_CONCURRENCY}" "${CPUS_PER_TASK}"
printf 'finalize_job=%s\n' "${FINAL_JOB}"
printf 'labels_root=%s\n' "${LABELS_ROOT}"
