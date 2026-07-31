#!/usr/bin/env bash
# Submit 7 Skill-It DataDecide-60M one-hot probes as one Slurm array (1 GPU per task).
#
# Prereq: CPU prep finished (prepare_probes.sh) with slices under RUN_DIR/slices/.
#
# Usage (FarmShare login node):
#   RUN_DIR=/scratch/users/$USER/agent-runs/skillit-probes-YYYYMMDD-HHMMSS \
#     bash experiments/skill-dag/skillit/submit_skillit_probes.sh
#
# Optional:
#   MAX_PARALLEL=4   cap concurrent running tasks (default: unset = no cap;
#                    each task starts independently when any GPU is free)
#   ARRAY_TASKS=0-6  override task range
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/skillit-probes-20260729-112123}"
EDULLM_ROOT="${EDULLM_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
LADDER_RUN="${LADDER_RUN:-/scratch/users/${SUNET}/agent-runs/olmo-ladder-370m-20260722-185217}"
SKILLIT_ROOT="${SKILLIT_ROOT:-${EDULLM_ROOT}/experiments/skill-dag/skillit}"
MIXLAW_ROOT="${MIXLAW_ROOT:-${EDULLM_ROOT}/experiments/skill-dag/mixlaw}"
VENV="${VENV:-${LADDER_RUN}/venv}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
ARRAY_TASKS="${ARRAY_TASKS:-0-6}"
RESULTS_S3="${RESULTS_S3:-s3://edullm-checkpoints/skillit/probes}"

if [[ ! -d "${RUN_DIR}/slices" ]]; then
  echo "missing slices under ${RUN_DIR}/slices (run submit_skillit_prepare_probes.sh first)" >&2
  exit 2
fi
if [[ ! -f "${SKILLIT_ROOT}/skillit_probe.sbatch" ]]; then
  echo "missing ${SKILLIT_ROOT}/skillit_probe.sbatch (sync repo to ${EDULLM_ROOT} first)" >&2
  exit 2
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "missing GPU venv at ${VENV}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/runs"

# Stable probe order (matches probes.json mixtures[].run_name).
cat > "${RUN_DIR}/probe_ids.txt" <<'EOF'
probe_dclm
probe_arxiv
probe_starcoder
probe_pes2o
probe_open-web-math
probe_algebraic-stack
probe_wiki
EOF

N_PROBES="$(wc -l < "${RUN_DIR}/probe_ids.txt" | tr -d ' ')"
LAST_IDX=$((N_PROBES - 1))
if [[ "${ARRAY_TASKS}" == "0-6" && "${N_PROBES}" -ne 7 ]]; then
  ARRAY_TASKS="0-${LAST_IDX}"
fi

for probe in $(cat "${RUN_DIR}/probe_ids.txt"); do
  if [[ ! -f "${RUN_DIR}/slices/${probe}/paths_train.txt" ]]; then
    echo "missing ${RUN_DIR}/slices/${probe}/paths_train.txt" >&2
    exit 2
  fi
done

cat >> "${RUN_DIR}/env.sh" <<EOF

# GPU probe array (appended $(date -Is))
LADDER_RUN=${LADDER_RUN}
RESULTS_S3=${RESULTS_S3}
PROBE_ARRAY_TASKS=${ARRAY_TASKS}
PROBE_MAX_PARALLEL=${MAX_PARALLEL}
EOF

ARRAY_SPEC="${ARRAY_TASKS}"
if [[ -n "${MAX_PARALLEL}" && "${MAX_PARALLEL}" != "0" ]]; then
  ARRAY_SPEC="${ARRAY_TASKS}%${MAX_PARALLEL}"
fi

PROBE_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --array="${ARRAY_SPEC}" \
  --job-name=skillit-probe \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",SKILLIT_ROOT="${SKILLIT_ROOT}",MIXLAW_ROOT="${MIXLAW_ROOT}",LADDER_RUN="${LADDER_RUN}",VENV="${VENV}",RESULTS_S3="${RESULTS_S3}" \
  "${SKILLIT_ROOT}/skillit_probe.sbatch")

echo "probe_array_job_id=${PROBE_JOB}"
echo "${PROBE_JOB}" > "${RUN_DIR}/probe_array_job_id.txt"
echo "RUN_DIR=${RUN_DIR}"
echo "array=${ARRAY_SPEC} (1 GPU per task; tasks schedule independently)"
echo "submitted skillit-probe array=${PROBE_JOB}"
