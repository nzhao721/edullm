#!/usr/bin/env bash
# Submit max-parallel labeling for the RegMix 10B trimmed corpus.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
: "${RUN_DIR:=/scratch/users/${SUNET}/agent-runs/regmix-10b-20260725-124810}"
: "${VENV:=${RUN_DIR}/venv}"
: "${LABELS_ROOT:=${RUN_DIR}/labels}"
: "${ARRAY_CONCURRENCY:=7}"
: "${CPUS_PER_TASK:=16}"
: "${MEM:=48G}"
: "${TIME_LIMIT:=24:00:00}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${LABELS_ROOT}"

OLMO_DIR=""
if [[ -d "${SCRIPT_DIR}/../olmo" ]]; then
  OLMO_DIR="$(cd "${SCRIPT_DIR}/../olmo" && pwd)"
fi

if [[ "${SCRIPT_DIR}" != "${RUN_DIR}/scripts" ]]; then
  for name in text_difficulty_metrics.py label_olmo_shard.py finalize_olmo_labels.py \
    materialize_curriculum.py build_regmix_label_manifest.py label_regmix_shard.sbatch; do
    src="${SCRIPT_DIR}/${name}"
    if [[ ! -f "${src}" && -n "${OLMO_DIR}" ]]; then
      src="${OLMO_DIR}/${name}"
    fi
    if [[ ! -f "${src}" ]]; then
      echo "missing script: ${name}" >&2
      exit 1
    fi
    cp -f "${src}" "${RUN_DIR}/scripts/${name}"
  done
fi
chmod +x "${RUN_DIR}/scripts/"*.py "${RUN_DIR}/scripts/"*.sbatch 2>/dev/null || true

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "missing venv at ${VENV}; create one in the RegMix run dir first" >&2
  exit 1
fi

source "${VENV}/bin/activate"
export PYTHONPATH="${RUN_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

MANIFEST="${LABELS_ROOT}/label_manifest.jsonl"
python "${RUN_DIR}/scripts/build_regmix_label_manifest.py" --run-dir "${RUN_DIR}" --out "${MANIFEST}"
N="$(wc -l < "${MANIFEST}" | tr -d ' ')"
if [[ "${N}" -lt 1 ]]; then
  echo "empty label manifest" >&2
  exit 1
fi

ARRAY_MAX=$((N - 1))
if [[ "${ARRAY_CONCURRENCY}" -gt "${N}" ]]; then
  ARRAY_CONCURRENCY="${N}"
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
  "${RUN_DIR}/scripts/label_regmix_shard.sbatch")"

FINAL_JOB="$(sbatch --parsable \
  --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task=4 \
  --mem=16G \
  --time=06:00:00 \
  --job-name=regmix-label-final \
  --dependency="afterok:${LABEL_JOB}" \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/label-finalize-%j.out" \
  --error="${RUN_DIR}/logs/label-finalize-%j.err" \
  --wrap="source '${VENV}/bin/activate'; export PYTHONPATH='${RUN_DIR}/scripts'; python '${RUN_DIR}/scripts/finalize_olmo_labels.py' --labels-root '${LABELS_ROOT}' --manifest '${MANIFEST}' --corpus 'RegMix-optimized OLMo-mix 10B'")"

cat >"${LABELS_ROOT}/jobs.json" <<EOF
{
  "label_array_job": "${LABEL_JOB}",
  "finalize_job": "${FINAL_JOB}",
  "n_shards": ${N},
  "array_concurrency": ${ARRAY_CONCURRENCY},
  "cpus_per_task": ${CPUS_PER_TASK},
  "labels_root": "${LABELS_ROOT}",
  "run_dir": "${RUN_DIR}",
  "wheat_01_excluded": true
}
EOF

printf 'label_array_job=%s (%s tasks, concurrency=%s, cpus/task=%s)\n' \
  "${LABEL_JOB}" "${N}" "${ARRAY_CONCURRENCY}" "${CPUS_PER_TASK}"
printf 'finalize_job=%s\n' "${FINAL_JOB}"
printf 'labels_root=%s\n' "${LABELS_ROOT}"
