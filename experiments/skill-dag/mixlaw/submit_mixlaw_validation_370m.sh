#!/usr/bin/env bash
# Submit OLMo2-370M CE training for every mix in validation_mixtures_10b.json.
#
# Ephemeral / empty-scratch friendly:
#   - Stages (or reuses) a working pool from published s3://edullm-data/
#   - Never reads s3://edullm-datasets/
#   - Does not assume a persistent scratch pool, checkpoint tree, or ladder venv
#   - Durable artifacts: trainer fail-closed syncs to
#       s3://edullm-checkpoints/mixlaw/370m-validation/<mix>/
#     (missing aws/creds or sync failure aborts the job). Opt out: S3_EXPORT=0
#   - train_one.sh also fail-closed syncs SAVE/PROGRESS before exit
#
# Required:
#   TRAIN_VENV       Absolute path to a Python env with torch + olmo-core (+ mixlaw
#                    imports). Provide a prebuilt GPU image/env — this script does
#                    not install CUDA torch. (Alias: LADDER_VENV for older callers.)
#
# Optional:
#   SAVE_ROOT        checkpoint parent (default: ${RUN_DIR}/save)
#   PROGRESS_ROOT    progress parent (default: ${RUN_DIR}/progress)
#   POOL_DIR         pre-staged working pool; else stage into ${RUN_DIR}/pool
#   DATASET_ID       default pretrain/olmo-127b
#   DATASET_VERSION  pin; default resolve_latest
#   RECOVERY_MODE    fresh|resume|fail (default fail; per-arm resume needs
#                    EXTRA_ARGS with --load-path or staged durable metadata)
#   LADDER_BASE_CONFIG required OLMo2-370M ladder config for strict eval
#   RECIPE           path to validation_mixtures_10b.json
#   NPROC            GPUs per mix (default 1)
#   ARRAY_TASKS      override Slurm array (default 0..(n_mixes-1))
#   RUN_DIR          ephemeral run root (default: /scratch/.../mixlaw-370m-val-<ts>)
#   EDULLM_ROOT      only for optional FarmShare AWS session helpers (default: repo)
#   POOL_VENV        env for staging (default: create RUN_DIR/pool-venv)
#   S3_EXPORT        0 to disable live aws s3 sync (default: enabled)
#   WANDB_MODE       online|offline|disabled (default: online if
#                    ${RUN_DIR}/wandb-session.env present)
#   WANDB_PROJECT    default mixlaw
#
# Optional W&B (additive to fail-closed S3):
#   bash scripts/farmshare/push_wandb_session_to_farmshare.sh "$RUN_DIR"
set -Eeuo pipefail

SUNET="${SUNET:-${USER:-nzhao2}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EDULLM_ROOT="${EDULLM_ROOT:-${REPO_ROOT}}"
RECIPE="${RECIPE:-${SCRIPT_DIR}/validation_mixtures_10b.json}"
DATASET_ID="${DATASET_ID:-pretrain/olmo-127b}"
EDULLM_DATA_PKG="${EDULLM_DATA_PKG:-edullm-data @ git+https://github.com/edu-llm/edullm-data@main}"

TRAIN_VENV="${TRAIN_VENV:-${LADDER_VENV:-}}"
if [[ -z "${TRAIN_VENV}" ]]; then
  echo "Set TRAIN_VENV to a GPU Python env with torch + olmo-core (no hardcoded ladder path)." >&2
  echo "Example: TRAIN_VENV=/path/to/venv bash $0" >&2
  exit 2
fi
if [[ ! -x "${TRAIN_VENV}/bin/python" ]]; then
  echo "missing TRAIN_VENV python at ${TRAIN_VENV}/bin/python" >&2
  exit 2
fi

NPROC="${NPROC:-1}"
RUN_NAME="${RUN_NAME:-mixlaw-370m-val-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
SAVE_ROOT="${SAVE_ROOT:-${RUN_DIR}/save}"
PROGRESS_ROOT="${PROGRESS_ROOT:-${RUN_DIR}/progress}"
BUILD_WORKERS="${BUILD_WORKERS:-4}"
S3_EXPORT="${S3_EXPORT:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-mixlaw}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-370m-validation}"
RECOVERY_MODE="${RECOVERY_MODE:-fail}"
: "${LADDER_BASE_CONFIG:?Set LADDER_BASE_CONFIG to the OLMo2-370M ladder config}"
if [[ -f "${RUN_DIR}/wandb-session.env" ]]; then
  WANDB_MODE="${WANDB_MODE:-online}"
else
  WANDB_MODE="${WANDB_MODE:-disabled}"
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/work" "${SAVE_ROOT}" "${PROGRESS_ROOT}"
cp -a "${RECIPE}" "${RUN_DIR}/validation_mixtures_10b.json"
cp -a "${SCRIPT_DIR}/prepare_validation_370m_data.py" \
  "${SCRIPT_DIR}/launch_validation_370m.sh" \
  "${SCRIPT_DIR}/train_mixlaw_validation_370m.py" \
  "${SCRIPT_DIR}/mixlaw_runtime.py" \
  "${SCRIPT_DIR}/preflight_validation_370m.py" \
  "${SCRIPT_DIR}/mixlaw_wandb.py" \
  "${SCRIPT_DIR}/stage_validation_pool_from_edullm_data.py" \
  "${SCRIPT_DIR}/domain_stream.py" \
  "${SCRIPT_DIR}/mixlaw_common.py" \
  "${RUN_DIR}/"

# Optional FarmShare AWS session helpers (never required for code layout).
if [[ -f "${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" ]]; then
  cp -a "${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" \
    "${EDULLM_ROOT}/scripts/farmshare/write_aws_session_env.py" \
    "${RUN_DIR}/" 2>/dev/null || true
fi
sed -i 's/\r$//' "${RUN_DIR}/"*.{sh,py} 2>/dev/null || true

# Light venv for recipe parse + edullm-data staging (not the GPU train env).
POOL_VENV="${POOL_VENV:-${RUN_DIR}/pool-venv}"
if [[ ! -x "${POOL_VENV}/bin/python" ]]; then
  python3 -m venv "${POOL_VENV}"
  # shellcheck disable=SC1091
  source "${POOL_VENV}/bin/activate"
  pip install -U pip wheel
  # shellcheck disable=SC2086
  pip install boto3 numpy ${EDULLM_DATA_PKG}
  deactivate || true
fi

mapfile -t MIX_NAMES < <(
  "${POOL_VENV}/bin/python" - <<'PY' "${RUN_DIR}/validation_mixtures_10b.json"
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

RESOLVE_ARGS=(--dataset-id "${DATASET_ID}")
if [[ -n "${DATASET_VERSION:-}" ]]; then
  RESOLVE_ARGS+=(--dataset-version "${DATASET_VERSION}")
fi

# Sidecars only here (no byte download). Version resolve needs AWS + edullm-data.
# Prefer skip-resolve on the login node if AWS session is unavailable; the pool
# job (or trainer auto-stage) will resolve when credentials are present.
if [[ -f "${RUN_DIR}/prepare_aws_session_light.sh" ]]; then
  export EDULLM_ROOT RUN_DIR
  unset PREFIX || true
  # shellcheck disable=SC1091
  source "${RUN_DIR}/prepare_aws_session_light.sh" || true
  if [[ -n "${AWS_SESSION_ENV:-}" && -f "${AWS_SESSION_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${AWS_SESSION_ENV}" || true
  fi
fi
if ! "${POOL_VENV}/bin/python" "${RUN_DIR}/prepare_validation_370m_data.py" \
  --recipe "${RUN_DIR}/validation_mixtures_10b.json" \
  --work "${RUN_DIR}/work" \
  "${RESOLVE_ARGS[@]}"; then
  echo "prepare with resolve failed; retrying with --skip-resolve" >&2
  "${POOL_VENV}/bin/python" "${RUN_DIR}/prepare_validation_370m_data.py" \
    --recipe "${RUN_DIR}/validation_mixtures_10b.json" \
    --work "${RUN_DIR}/work" \
    --skip-resolve \
    "${RESOLVE_ARGS[@]}"
fi

printf '%s\n' "${MIX_NAMES[@]}" > "${RUN_DIR}/mix_names.txt"

DEP_OPTS=()
if [[ -z "${POOL_DIR:-}" ]]; then
  POOL_DIR="${RUN_DIR}/pool"
  mkdir -p "${POOL_DIR}"

  VERSION_ARG=""
  if [[ -n "${DATASET_VERSION:-}" ]]; then
    VERSION_ARG="--dataset-version ${DATASET_VERSION}"
  fi

  AWS_BOOT=""
  if [[ -f "${RUN_DIR}/prepare_aws_session_light.sh" ]]; then
    AWS_BOOT="export EDULLM_ROOT=${EDULLM_ROOT} RUN_DIR=${RUN_DIR}; source ${RUN_DIR}/prepare_aws_session_light.sh; source \${AWS_SESSION_ENV};"
  fi

  POOL_JOB=$(sbatch --parsable --exclude=wheat-01 \
    --partition=normal \
    --cpus-per-task="${BUILD_WORKERS}" \
    --mem=128G \
    --time=12:00:00 \
    --job-name=mixlaw-pool \
    --chdir="${RUN_DIR}" \
    --output="${RUN_DIR}/logs/pool-%j.out" \
    --error="${RUN_DIR}/logs/pool-%j.err" \
    --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; export PATH=\"\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}\"; ${AWS_BOOT} source ${POOL_VENV}/bin/activate; python ${RUN_DIR}/stage_validation_pool_from_edullm_data.py --out-dir ${POOL_DIR} --mixtures-json ${RUN_DIR}/validation_mixtures_10b.json --budget-tokens 10000000000 --dataset-id ${DATASET_ID} ${VERSION_ARG}'")
  echo "pool_job_id=${POOL_JOB}"
  echo "${POOL_JOB}" > "${RUN_DIR}/pool_job_id.txt"
  DEP_OPTS=(--dependency="afterok:${POOL_JOB}")
fi

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
EDULLM_ROOT=${EDULLM_ROOT}
TRAIN_VENV=${TRAIN_VENV}
NPROC=${NPROC}
POOL_DIR=${POOL_DIR}
DATASET_ID=${DATASET_ID}
DATASET_VERSION=${DATASET_VERSION:-}
SAVE_ROOT=${SAVE_ROOT}
PROGRESS_ROOT=${PROGRESS_ROOT}
S3_EXPORT=${S3_EXPORT}
WANDB_PROJECT=${WANDB_PROJECT}
WANDB_ENTITY=${WANDB_ENTITY}
WANDB_GROUP=${WANDB_GROUP}
WANDB_MODE=${WANDB_MODE}
RECOVERY_MODE=${RECOVERY_MODE}
LADDER_BASE_CONFIG=${LADDER_BASE_CONFIG}
# Ephemeral RUN_DIR copies trainers; curriculum/token-selection stay in the checkout.
PYTHONPATH=${RUN_DIR}:${EDULLM_ROOT}/experiments/curriculum:${EDULLM_ROOT}/experiments/token-selection
export PYTHONPATH EDULLM_ROOT LADDER_BASE_CONFIG
EOF

cat > "${RUN_DIR}/train_one.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
source "${RUN_DIR}/env.sh"
# shellcheck disable=SC1091
source "${TRAIN_VENV}/bin/activate"
IDX="${SLURM_ARRAY_TASK_ID:?}"
MIX_NAME="$(sed -n "$((IDX + 1))p" "${RUN_DIR}/mix_names.txt")"
: "${MIX_NAME:?empty MIX_NAME for index ${IDX}}"
export MIX_NAME
export POOL_DIR
export DATASET_ID
export DATASET_VERSION
export MIX_WEIGHTS_JSON="${RUN_DIR}/work/${MIX_NAME}/mix_weights.json"
export SAVE_FOLDER="${SAVE_ROOT}/${MIX_NAME}/checkpoints"
export PROGRESS_DIR="${PROGRESS_ROOT}/${MIX_NAME}/progress"
export NPROC S3_EXPORT
export EDULLM_ROOT PYTHONPATH
export WANDB_PROJECT WANDB_ENTITY WANDB_GROUP WANDB_MODE
export RECOVERY_MODE LADDER_BASE_CONFIG
export RUN_DIR
# Pool staged by dependency job (or provided); refuse GPU-job re-fetch.
# Do not auto-pass --fresh: empty scratch starts clean; leftover local ckpts
# fail closed (pass EXTRA_ARGS='--load-path …' or '--fresh').
export EXTRA_ARGS="${EXTRA_ARGS:-} --no-auto-stage"
mkdir -p "${SAVE_FOLDER}" "${PROGRESS_DIR}"
if [[ -f "${RUN_DIR}/wandb-session.env" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${RUN_DIR}/wandb-session.env"
  set -u
fi
bash "${RUN_DIR}/launch_validation_370m.sh"
# Upload-before-end belt-and-suspenders (fail-closed when S3_EXPORT enabled).
if [[ "${S3_EXPORT}" == "0" || "${S3_EXPORT}" == "false" || "${S3_EXPORT}" == "no" || "${S3_EXPORT}" == "off" ]]; then
  echo "[train_one] S3_EXPORT=${S3_EXPORT}: skip upload-before-end (local smoke)" >&2
else
  if ! command -v aws >/dev/null 2>&1; then
    echo "[train_one] aws CLI missing; refuse upload-before-end (set S3_EXPORT=0 for smoke)" >&2
    exit 2
  fi
  aws s3 sync "${SAVE_FOLDER}" \
    "s3://edullm-checkpoints/mixlaw/370m-validation/${MIX_NAME}/checkpoints/" \
    --only-show-errors
  aws s3 sync "${PROGRESS_DIR}" \
    "s3://edullm-checkpoints/mixlaw/370m-validation/${MIX_NAME}/progress/" \
    --only-show-errors
  TL_DIR="${PROGRESS_ROOT}/${MIX_NAME}/task_loss_results"
  if [[ -d "${TL_DIR}" ]]; then
    aws s3 sync "${TL_DIR}" \
      "s3://edullm-checkpoints/mixlaw/370m-validation/${MIX_NAME}/task_loss_results/" \
      --only-show-errors
  fi
fi
EOF
chmod +x "${RUN_DIR}/train_one.sh"

JOB=$(sbatch --parsable --exclude=wheat-01 \
  "${DEP_OPTS[@]}" \
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
echo "POOL_DIR=${POOL_DIR}"
echo "DATASET_ID=${DATASET_ID}"
echo "TRAIN_VENV=${TRAIN_VENV}"
echo "SAVE_ROOT=${SAVE_ROOT}"
echo "PROGRESS_ROOT=${PROGRESS_ROOT}"
echo "S3_EXPORT=${S3_EXPORT}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "mixes (${N_MIXES}): ${MIX_NAMES[*]}"
echo "${JOB}" > "${RUN_DIR}/job_id.txt"
