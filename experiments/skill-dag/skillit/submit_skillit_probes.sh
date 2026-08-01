#!/usr/bin/env bash
# Submit 7 Skill-It DataDecide-60M one-hot probes as one Slurm array (1 GPU per task).
#
# Prereq: edullm-data pool + recipe sidecars (submit_skillit_prepare_probes.sh).
# Never assumes the old ladder scratch tree or s3://edullm-datasets/.
#
# Required:
#   TRAIN_VENV   GPU Python env with torch + olmo-core (alias: VENV / LADDER_VENV)
#   POOL_DIR     working pool with edullm_data_source.json
#
# Optional:
#   RUN_DIR      ephemeral run root (default: /scratch/.../skillit-probes-<ts>)
#   WANDB_PROJECT default skillit
#   WANDB_MODE    production requires online
#   ALLOW_LOCAL_ONLY 1 permits disabled/offline W&B for local smoke only
#
# Usage:
#   TRAIN_VENV=/path/to/venv POOL_DIR=$RUN_DIR/pool \
#     bash experiments/skill-dag/skillit/submit_skillit_probes.sh
#
# Optional W&B (SmolLM-style):
#   bash scripts/farmshare/push_wandb_session_to_farmshare.sh "$RUN_DIR"
set -Eeuo pipefail

SUNET="${SUNET:-${USER:?set SUNET or USER}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIXLAW_ROOT="${MIXLAW_ROOT:-$(cd "${SCRIPT_DIR}/../mixlaw" && pwd)}"

RUN_NAME="${RUN_NAME:-skillit-probes-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
POOL_DIR="${POOL_DIR:-}"
SKILLIT_ROOT="${SKILLIT_ROOT:-${SCRIPT_DIR}}"
TRAIN_VENV="${TRAIN_VENV:-${VENV:-${LADDER_VENV:-}}}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
ARRAY_TASKS="${ARRAY_TASKS:-0-6}"
RECIPE_WORK="${RECIPE_WORK:-${RUN_DIR}/recipe}"
DATASET_ID="${DATASET_ID:-pretrain/olmo-127b}"
DATASET_VERSION="${DATASET_VERSION:-v1}"

: "${POOL_DIR:?Set POOL_DIR to an edullm-data staged working pool root}"
if [[ -z "${TRAIN_VENV}" ]]; then
  echo "Set TRAIN_VENV to a GPU Python env with torch + olmo-core (no hardcoded ladder path)." >&2
  exit 2
fi
if [[ ! -x "${TRAIN_VENV}/bin/python" ]]; then
  echo "missing GPU venv at ${TRAIN_VENV}" >&2
  exit 2
fi
"${TRAIN_VENV}/bin/python" "${SKILLIT_ROOT}/prepare_skillit_370m_data.py" \
  --pool-dir "${POOL_DIR}" \
  --dataset-id "${DATASET_ID}" \
  --dataset-version "${DATASET_VERSION}" \
  --pool-layout probe \
  --validate-pool-only
if [[ ! -d "${RECIPE_WORK}" ]]; then
  echo "missing recipe sidecars under ${RECIPE_WORK} (run submit_skillit_prepare_probes.sh first)" >&2
  exit 2
fi
if [[ ! -f "${SKILLIT_ROOT}/skillit_probe.sbatch" ]]; then
  echo "missing ${SKILLIT_ROOT}/skillit_probe.sbatch (sync repo first)" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/runs"

WANDB_PROJECT="${WANDB_PROJECT:-skillit}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
ALLOW_LOCAL_ONLY="${ALLOW_LOCAL_ONLY:-0}"
if [[ -f "${RUN_DIR}/wandb-session.env" ]]; then
  WANDB_MODE="${WANDB_MODE:-online}"
else
  WANDB_MODE="${WANDB_MODE:-disabled}"
fi
if [[ "${ALLOW_LOCAL_ONLY}" != "1" && "${WANDB_MODE}" != "online" ]]; then
  echo "production requires WANDB_MODE=online; set ALLOW_LOCAL_ONLY=1 only for local smoke" >&2
  exit 2
fi

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
  if [[ ! -f "${RECIPE_WORK}/${probe}/mix_weights.json" ]]; then
    echo "missing ${RECIPE_WORK}/${probe}/mix_weights.json" >&2
    exit 2
  fi
done

cat >> "${RUN_DIR}/env.sh" <<EOF

# GPU probe array (appended $(date -Is 2>/dev/null || date))
TRAIN_VENV=${TRAIN_VENV}
POOL_DIR=${POOL_DIR}
RECIPE_WORK=${RECIPE_WORK}
PROBE_ARRAY_TASKS=${ARRAY_TASKS}
PROBE_MAX_PARALLEL=${MAX_PARALLEL}
WANDB_PROJECT=${WANDB_PROJECT}
WANDB_ENTITY=${WANDB_ENTITY}
WANDB_MODE=${WANDB_MODE}
ALLOW_LOCAL_ONLY=${ALLOW_LOCAL_ONLY}
DATASET_ID=${DATASET_ID}
DATASET_VERSION=${DATASET_VERSION}
EOF

ARRAY_SPEC="${ARRAY_TASKS}"
if [[ -n "${MAX_PARALLEL}" && "${MAX_PARALLEL}" != "0" ]]; then
  ARRAY_SPEC="${ARRAY_TASKS}%${MAX_PARALLEL}"
fi

PROBE_JOB=$(sbatch --parsable \
  --array="${ARRAY_SPEC}" \
  --job-name=skillit-probe \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",SKILLIT_ROOT="${SKILLIT_ROOT}",MIXLAW_ROOT="${MIXLAW_ROOT}",TRAIN_VENV="${TRAIN_VENV}",VENV="${TRAIN_VENV}",POOL_DIR="${POOL_DIR}",RECIPE_WORK="${RECIPE_WORK}",WANDB_PROJECT="${WANDB_PROJECT}",WANDB_ENTITY="${WANDB_ENTITY}",WANDB_MODE="${WANDB_MODE}",ALLOW_LOCAL_ONLY="${ALLOW_LOCAL_ONLY}",DATASET_ID="${DATASET_ID}",DATASET_VERSION="${DATASET_VERSION}" \
  "${SKILLIT_ROOT}/skillit_probe.sbatch")

echo "probe_array_job_id=${PROBE_JOB}"
echo "${PROBE_JOB}" > "${RUN_DIR}/probe_array_job_id.txt"
echo "RUN_DIR=${RUN_DIR} POOL_DIR=${POOL_DIR}"
echo "TRAIN_VENV=${TRAIN_VENV}"
echo "WANDB_PROJECT=${WANDB_PROJECT} WANDB_MODE=${WANDB_MODE}"
echo "array=${ARRAY_SPEC}"
