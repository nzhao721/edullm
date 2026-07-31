#!/usr/bin/env bash
# Submit OLMo2-370M CE training for every mix in validation_mixtures_10b.json.
#
# Domain weights are fixed by the already-built corpora under
#   s3://edullm-datasets/mixlaw/mixes/<run_name>/
# Rebuild those corpora first if the recipe changed
# (submit_mixlaw_validation_10b.sh).
#
# Required env:
#   TOKENIZED_ROOT   local root with one subdir per mix (synced from S3)
#   SAVE_ROOT        checkpoint parent (one subdir per mix)
#   PROGRESS_ROOT    progress parent (one subdir per mix)
#
# Optional:
#   RECIPE           path to validation_mixtures_10b.json
#   NPROC            GPUs per mix (default 1)
#   ARRAY_TASKS      override Slurm array (default 0..(n_mixes-1))
#   LADDER_VENV      python/torch env for the control trainer
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDULLM_ROOT="${EDULLM_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
RECIPE="${RECIPE:-${SCRIPT_DIR}/validation_mixtures_10b.json}"
: "${TOKENIZED_ROOT:?Set TOKENIZED_ROOT to local mixlaw mixes root}"
: "${SAVE_ROOT:?Set SAVE_ROOT}"
: "${PROGRESS_ROOT:?Set PROGRESS_ROOT}"

NPROC="${NPROC:-1}"
LADDER_VENV="${LADDER_VENV:-/scratch/users/${SUNET}/agent-runs/olmo-ladder-370m-20260722-185217/venv}"
RUN_NAME="${RUN_NAME:-mixlaw-370m-val-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/work" "${SAVE_ROOT}" "${PROGRESS_ROOT}"
cp -a "${RECIPE}" "${RUN_DIR}/validation_mixtures_10b.json"
cp -a "${SCRIPT_DIR}/prepare_validation_370m_data.py" \
  "${SCRIPT_DIR}/launch_validation_370m.sh" \
  "${SCRIPT_DIR}/mixlaw_common.py" \
  "${RUN_DIR}/"

# Resolve mix list from the recipe (source of truth for domain weights / names).
mapfile -t MIX_NAMES < <(
  "${LADDER_VENV}/bin/python" - <<'PY' "${RUN_DIR}/validation_mixtures_10b.json"
import json, sys
recipe = json.loads(open(sys.argv[1], encoding="utf-8").read())
for m in recipe["mixtures"]:
    print(m["run_name"])
PY
)
N_MIXES="${#MIX_NAMES[@]}"
if [[ "${N_MIXES}" -lt 1 ]]; then
  echo "no mixtures in ${RECIPE}" >&2
  exit 1
fi
ARRAY_TASKS="${ARRAY_TASKS:-0-$((N_MIXES - 1))}"

# Materialize paths_train.txt + mix_weights.json for every recipe arm.
"${LADDER_VENV}/bin/python" "${RUN_DIR}/prepare_validation_370m_data.py" \
  --recipe "${RUN_DIR}/validation_mixtures_10b.json" \
  --tokenized-root "${TOKENIZED_ROOT}" \
  --work "${RUN_DIR}/work"

printf '%s\n' "${MIX_NAMES[@]}" > "${RUN_DIR}/mix_names.txt"

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
EDULLM_ROOT=${EDULLM_ROOT}
LADDER_VENV=${LADDER_VENV}
NPROC=${NPROC}
SAVE_ROOT=${SAVE_ROOT}
PROGRESS_ROOT=${PROGRESS_ROOT}
EOF

cat > "${RUN_DIR}/train_one.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
source "${RUN_DIR}/env.sh"
# shellcheck disable=SC1091
source "${LADDER_VENV}/bin/activate"
IDX="${SLURM_ARRAY_TASK_ID:?}"
MIX_NAME="$(sed -n "$((IDX + 1))p" "${RUN_DIR}/mix_names.txt")"
: "${MIX_NAME:?empty MIX_NAME for index ${IDX}}"
export MIX_NAME
export TRAIN_PATHS_FILE="${RUN_DIR}/work/${MIX_NAME}/paths_train.txt"
export SAVE_FOLDER="${SAVE_ROOT}/${MIX_NAME}/checkpoints"
export PROGRESS_DIR="${PROGRESS_ROOT}/${MIX_NAME}/progress"
export NPROC
mkdir -p "${SAVE_FOLDER}" "${PROGRESS_DIR}"
bash "${RUN_DIR}/launch_validation_370m.sh"
EOF
chmod +x "${RUN_DIR}/train_one.sh"

JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=gpu \
  --qos=gpu \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=0 \
  --gres=gpu:"${NPROC}" \
  --time=48:00:00 \
  --array="${ARRAY_TASKS}" \
  --job-name=mixlaw-370m \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/train-%A_%a.out" \
  --error="${RUN_DIR}/logs/train-%A_%a.err" \
  --export=ALL,RUN_DIR="${RUN_DIR}" \
  --wrap="bash ${RUN_DIR}/train_one.sh")

echo "job_id=${JOB}"
echo "RUN_DIR=${RUN_DIR}"
echo "mixes (${N_MIXES}): ${MIX_NAMES[*]}"
echo "${JOB}" > "${RUN_DIR}/job_id.txt"
