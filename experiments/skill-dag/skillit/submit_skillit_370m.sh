#!/usr/bin/env bash
# Submit Skill-It OLMo2-370M dual-arm training (probe + derivative).
#
# Ephemeral / empty-scratch friendly:
#   - Stages (or reuses) a working pool from published s3://edullm-data/
#   - Never reads s3://edullm-datasets/
#   - Does not assume a persistent scratch pool, checkpoint tree, or ladder venv
#   - Checkpoints/progress/evals stay on runtime scratch and upload to W&B
#   - S3 is used only to stage training data or an explicit resume bootstrap
#
# Required:
#   TRAIN_VENV       Absolute path to a Python env with torch + olmo-core (+ skillit
#                    imports). Provide a prebuilt GPU image/env — this script does
#                    not install CUDA torch. (Alias: LADDER_VENV for older callers.)
#
# Optional:
#   SAVE_ROOT        checkpoint parent (default: ${RUN_DIR}/save)
#   PROGRESS_ROOT    progress parent (default: ${RUN_DIR}/progress)
#   POOL_DIR         pre-staged pool with _EDULLM_DATA_SOURCE.json;
#                    else stage into ${RUN_DIR}/pool via a CPU Slurm job
#   DATASET_ID       pinned pretrain/olmo-127b
#   DATASET_VERSION  pinned v1
#   RESUME_MODE      required: fresh | resume
#   LOAD_PATH_TEMPLATE resume path containing {arm_id}, local or S3 stepN
#   LADDER_BASE_CONFIG compatible OLMES YAML (required)
#   RECIPE           path to skillit_train_recipe.json
#   NPROC            GPUs per arm (default 1)
#   ARRAY_TASKS      override Slurm array (default 0-1)
#   RUN_DIR          ephemeral run root (default: /scratch/.../skillit-370m-<ts>)
#   EDULLM_ROOT      only for optional FarmShare AWS session helpers
#   WANDB_PROJECT    default skillit
#   WANDB_MODE       production requires online
#   ALLOW_LOCAL_ONLY 1 permits disabled/offline W&B for local smoke only
#   WANDB_ENTITY / WANDB_GROUP / WANDB_UPLOAD_EXISTING
#
# Mint/push optional W&B session from the laptop (SmolLM-style):
#   bash scripts/farmshare/push_wandb_session_to_farmshare.sh "$RUN_DIR"
set -Eeuo pipefail

SUNET="${SUNET:-${USER:?set SUNET or USER}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIXLAW_ROOT="$(cd "${SCRIPT_DIR}/../mixlaw" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EDULLM_ROOT="${EDULLM_ROOT:-${REPO_ROOT}}"
RECIPE="${RECIPE:-${SCRIPT_DIR}/skillit_train_recipe.json}"
DATASET_ID="${DATASET_ID:-pretrain/olmo-127b}"
DATASET_VERSION="${DATASET_VERSION:-v1}"
EDULLM_DATA_PKG="${EDULLM_DATA_PKG:-edullm-data @ git+https://github.com/edu-llm/edullm-data@main}"
RESUME_MODE="${RESUME_MODE:-}"
LOAD_PATH_TEMPLATE="${LOAD_PATH_TEMPLATE:-}"
LADDER_BASE_CONFIG="${LADDER_BASE_CONFIG:-}"

if [[ "${DATASET_ID}" != "pretrain/olmo-127b" || "${DATASET_VERSION}" != "v1" ]]; then
  echo "SkillIt source is pinned to pretrain/olmo-127b/v1" >&2
  exit 2
fi
case "${RESUME_MODE}" in
  fresh)
    [[ -z "${LOAD_PATH_TEMPLATE}" ]] || { echo "RESUME_MODE=fresh conflicts with LOAD_PATH_TEMPLATE" >&2; exit 2; }
    ;;
  resume)
    [[ "${LOAD_PATH_TEMPLATE}" == *"{arm_id}"* ]] || {
      echo "RESUME_MODE=resume requires LOAD_PATH_TEMPLATE containing {arm_id}" >&2
      exit 2
    }
    ;;
  *)
    echo "Set RESUME_MODE=fresh or RESUME_MODE=resume" >&2
    exit 2
    ;;
esac
if [[ -z "${LADDER_BASE_CONFIG}" || ! -f "${LADDER_BASE_CONFIG}" ]]; then
  echo "Set LADDER_BASE_CONFIG to an existing compatible OLMES YAML" >&2
  exit 2
fi

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
RUN_NAME="${RUN_NAME:-skillit-370m-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
SAVE_ROOT="${SAVE_ROOT:-${RUN_DIR}/save}"
PROGRESS_ROOT="${PROGRESS_ROOT:-${RUN_DIR}/progress}"
BUILD_WORKERS="${BUILD_WORKERS:-4}"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/work" "${SAVE_ROOT}" "${PROGRESS_ROOT}"
cp -a "${RECIPE}" "${RUN_DIR}/skillit_train_recipe.json"
cp -a "${SCRIPT_DIR}/prepare_skillit_370m_data.py" \
  "${SCRIPT_DIR}/launch_arm.sh" \
  "${SCRIPT_DIR}/train_skillit_370m.py" \
  "${SCRIPT_DIR}/skillit_math.py" \
  "${SCRIPT_DIR}/wandb_logging.py" \
  "${MIXLAW_ROOT}/domain_stream.py" \
  "${MIXLAW_ROOT}/mixlaw_common.py" \
  "${RUN_DIR}/"

mkdir -p "${RUN_DIR}/artifacts"
if [[ -f "${SCRIPT_DIR}/artifacts/probes_full/A_offline.npy" ]]; then
  cp -a "${SCRIPT_DIR}/artifacts/probes_full/A_offline.npy" "${RUN_DIR}/artifacts/A_offline.npy"
elif [[ -f "${SCRIPT_DIR}/artifacts/A_offline.npy" ]]; then
  cp -a "${SCRIPT_DIR}/artifacts/A_offline.npy" "${RUN_DIR}/artifacts/A_offline.npy"
else
  echo "warning: no A_offline.npy under ${SCRIPT_DIR}/artifacts — set A_OFFLINE for probe arm" >&2
fi
if [[ -f "${MIXLAW_ROOT}/mixlaw_fit_chinchilla.json" ]]; then
  cp -a "${MIXLAW_ROOT}/mixlaw_fit_chinchilla.json" "${RUN_DIR}/"
fi

if [[ -f "${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" ]]; then
  cp -a "${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" \
    "${EDULLM_ROOT}/scripts/farmshare/write_aws_session_env.py" \
    "${RUN_DIR}/" 2>/dev/null || true
fi
sed -i 's/\r$//' "${RUN_DIR}/"*.{sh,py} 2>/dev/null || true

# Light venv for recipe sidecars + edullm-data staging (not the GPU train env).
POOL_VENV="${POOL_VENV:-${RUN_DIR}/pool-venv}"
if [[ ! -x "${POOL_VENV}/bin/python" ]]; then
  python3 -m venv "${POOL_VENV}"
  # shellcheck disable=SC1091
  source "${POOL_VENV}/bin/activate"
  pip install -U pip wheel
  pip install boto3 numpy
  pip install --upgrade "${EDULLM_DATA_PKG}"
  deactivate || true
fi

# Arm sidecars do not need a staged pool (weights only).
"${POOL_VENV}/bin/python" - <<'PY' "${RUN_DIR}/skillit_train_recipe.json" "${RUN_DIR}/work" "${DATASET_ID}" "${DATASET_VERSION}"
import json, sys
from pathlib import Path

recipe = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
work = Path(sys.argv[2])
dataset_id = sys.argv[3]
dataset_version = sys.argv[4]
domains = recipe.get("domain_order") or [
    "dclm", "arxiv", "starcoder", "pes2o", "open-web-math", "algebraic-stack", "wiki"
]
if "initial_weights" in recipe:
    weights = {d: float(w) for d, w in zip(domains, recipe["initial_weights"])}
else:
    weights = {d: float(recipe["base_weights"][d]) for d in domains}
seed = int(recipe.get("seed", 42))
work.mkdir(parents=True, exist_ok=True)
arms = []
for arm in recipe["arms"]:
    arm_dir = work / arm["arm_id"]
    arm_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "arm_id": arm["arm_id"],
        "a_mode": arm["a_mode"],
        "id": arm["id"],
        "label": arm.get("label"),
        "domain_order": domains,
        "weights": weights,
        "recipe": "skillit_train_recipe.json",
        "recipe_seed": seed,
        "stream_seed": seed + int(arm["id"]),
        "budget_tokens": recipe.get("budget_tokens"),
        "edullm_data": {"dataset_id": dataset_id, "version": dataset_version},
        "sampling": "domain_stratified_stream",
        "skillit": recipe.get("skillit", {}),
    }
    path = arm_dir / "arm_weights.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    arms.append(
        {
            "arm_id": arm["arm_id"],
            "a_mode": arm["a_mode"],
            "id": arm["id"],
            "arm_weights": str(path.resolve()),
        }
    )
    print(arm["arm_id"])
(work / "skillit_arms.json").write_text(
    json.dumps({"dataset_id": dataset_id, "dataset_version": dataset_version, "arms": arms}, indent=2) + "\n",
    encoding="utf-8",
)
PY

mapfile -t ARM_IDS < <(
  "${POOL_VENV}/bin/python" - <<'PY' "${RUN_DIR}/skillit_train_recipe.json"
import json, sys
recipe = json.loads(open(sys.argv[1], encoding="utf-8").read())
for arm in recipe["arms"]:
    print(arm["arm_id"])
PY
)
N_ARMS="${#ARM_IDS[@]}"
if [[ "${N_ARMS}" -lt 1 ]]; then
  echo "no arms in ${RECIPE}" >&2
  exit 1
fi
ARRAY_TASKS="${ARRAY_TASKS:-0-$((N_ARMS - 1))}"
printf '%s\n' "${ARM_IDS[@]}" > "${RUN_DIR}/arm_ids.txt"

DEP_OPTS=()
if [[ -z "${POOL_DIR:-}" ]]; then
  POOL_DIR="${RUN_DIR}/pool"
  mkdir -p "${POOL_DIR}"
  AWS_BOOT=""
  if [[ -f "${RUN_DIR}/prepare_aws_session_light.sh" ]]; then
    AWS_BOOT="export EDULLM_ROOT=${EDULLM_ROOT} RUN_DIR=${RUN_DIR}; source ${RUN_DIR}/prepare_aws_session_light.sh; source \${AWS_SESSION_ENV};"
  fi
  VERSION_ARG=""
  if [[ -n "${DATASET_VERSION:-}" ]]; then
    VERSION_ARG="--dataset-version ${DATASET_VERSION}"
  fi
  POOL_JOB=$(sbatch --parsable \
    --partition=normal \
    --cpus-per-task="${BUILD_WORKERS}" \
    --mem=128G \
    --time=12:00:00 \
    --job-name=skillit-pool \
    --chdir="${RUN_DIR}" \
    --output="${RUN_DIR}/logs/pool-%j.out" \
    --error="${RUN_DIR}/logs/pool-%j.err" \
    --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; export PATH=\"\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}\"; ${AWS_BOOT} source ${POOL_VENV}/bin/activate; python ${RUN_DIR}/prepare_skillit_370m_data.py --recipe ${RUN_DIR}/skillit_train_recipe.json --work ${RUN_DIR}/work --pool-dir ${POOL_DIR} --dataset-id ${DATASET_ID} ${VERSION_ARG}'")
  echo "pool_job_id=${POOL_JOB}"
  echo "${POOL_JOB}" > "${RUN_DIR}/pool_job_id.txt"
  DEP_OPTS=(--dependency="afterok:${POOL_JOB}")
else
  "${POOL_VENV}/bin/python" "${RUN_DIR}/prepare_skillit_370m_data.py" \
    --pool-dir "${POOL_DIR}" \
    --dataset-id "${DATASET_ID}" \
    --dataset-version "${DATASET_VERSION}" \
    --validate-pool-only
fi

WANDB_PROJECT="${WANDB_PROJECT:-skillit}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-${RUN_NAME}}"
WANDB_UPLOAD_EXISTING="${WANDB_UPLOAD_EXISTING:-0}"
ALLOW_LOCAL_ONLY="${ALLOW_LOCAL_ONLY:-0}"
if [[ -f "${RUN_DIR}/wandb-session.env" ]]; then
  WANDB_MODE="${WANDB_MODE:-online}"
else
  WANDB_MODE="${WANDB_MODE:-disabled}"
fi
if [[ "${WANDB_MODE}" == "online" && ! -f "${RUN_DIR}/wandb-session.env" ]]; then
  echo "missing ${RUN_DIR}/wandb-session.env (required for WANDB_MODE=online)" >&2
  echo "push: bash scripts/farmshare/push_wandb_session_to_farmshare.sh ${RUN_DIR}" >&2
  exit 2
fi
if [[ "${ALLOW_LOCAL_ONLY}" != "1" && "${WANDB_MODE}" != "online" ]]; then
  echo "production requires WANDB_MODE=online; set ALLOW_LOCAL_ONLY=1 only for local smoke" >&2
  exit 2
fi

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
EDULLM_ROOT=${EDULLM_ROOT}
MIXLAW_ROOT=${EDULLM_ROOT}/experiments/skill-dag/mixlaw
TRAIN_VENV=${TRAIN_VENV}
NPROC=${NPROC}
POOL_DIR=${POOL_DIR}
DATASET_ID=${DATASET_ID}
DATASET_VERSION=${DATASET_VERSION:-}
RESUME_MODE=${RESUME_MODE}
LOAD_PATH_TEMPLATE=${LOAD_PATH_TEMPLATE}
LADDER_BASE_CONFIG=${LADDER_BASE_CONFIG}
SAVE_ROOT=${SAVE_ROOT}
PROGRESS_ROOT=${PROGRESS_ROOT}
ALLOW_LOCAL_ONLY=${ALLOW_LOCAL_ONLY}
A_OFFLINE=${RUN_DIR}/artifacts/A_offline.npy
MIXLAW_FIT_JSON=${RUN_DIR}/mixlaw_fit_chinchilla.json
WANDB_PROJECT=${WANDB_PROJECT}
WANDB_ENTITY=${WANDB_ENTITY}
WANDB_GROUP=${WANDB_GROUP}
WANDB_MODE=${WANDB_MODE}
WANDB_UPLOAD_EXISTING=${WANDB_UPLOAD_EXISTING}
# Ephemeral RUN_DIR copies trainers; curriculum/token-selection stay in the checkout.
PYTHONPATH=${RUN_DIR}:${EDULLM_ROOT}/experiments/skill-dag/mixlaw:${EDULLM_ROOT}/experiments/curriculum:${EDULLM_ROOT}/experiments/token-selection
export PYTHONPATH EDULLM_ROOT MIXLAW_ROOT
EOF

cat > "${RUN_DIR}/train_one.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
source "${RUN_DIR}/env.sh"
# shellcheck disable=SC1091
source "${TRAIN_VENV}/bin/activate"
set +u
if [[ -f "${RUN_DIR}/wandb-session.env" ]]; then
  # shellcheck disable=SC1090
  source "${RUN_DIR}/wandb-session.env"
fi
if [[ -f "${RUN_DIR}/aws-session.env" ]]; then
  # shellcheck disable=SC1090
  source "${RUN_DIR}/aws-session.env"
fi
if [[ -f "${RUN_DIR}/hf-session.env" ]]; then
  # shellcheck disable=SC1090
  source "${RUN_DIR}/hf-session.env"
fi
set -u
IDX="${SLURM_ARRAY_TASK_ID:?}"
ARM_ID="$(sed -n "$((IDX + 1))p" "${RUN_DIR}/arm_ids.txt")"
: "${ARM_ID:?empty ARM_ID for index ${IDX}}"
ARM_WEIGHTS_JSON="${RUN_DIR}/work/${ARM_ID}/arm_weights.json"
A_MODE="$("${TRAIN_VENV}/bin/python" -c "import json; print(json.load(open('${ARM_WEIGHTS_JSON}'))['a_mode'])")"
LOAD_PATH="${LOAD_PATH_TEMPLATE//\{arm_id\}/${ARM_ID}}"
export ARM_ID A_MODE POOL_DIR DATASET_ID DATASET_VERSION ARM_WEIGHTS_JSON
export RESUME_MODE LOAD_PATH LADDER_BASE_CONFIG
export SAVE_FOLDER="${SAVE_ROOT}/${ARM_ID}/checkpoints"
export PROGRESS_DIR="${PROGRESS_ROOT}/${ARM_ID}/progress"
export NPROC ALLOW_LOCAL_ONLY
export A_OFFLINE MIXLAW_FIT_JSON
export EDULLM_ROOT MIXLAW_ROOT PYTHONPATH
export WANDB_PROJECT WANDB_ENTITY WANDB_GROUP WANDB_MODE WANDB_UPLOAD_EXISTING
export WANDB_RUN_NAME="${ARM_ID}"
export WANDB_DIR="${PROGRESS_DIR}/wandb"
mkdir -p "${SAVE_FOLDER}" "${PROGRESS_DIR}" "${WANDB_DIR}"
"${TRAIN_VENV}/bin/python" "${RUN_DIR}/prepare_skillit_370m_data.py" \
  --pool-dir "${POOL_DIR}" \
  --dataset-id "${DATASET_ID}" \
  --dataset-version "${DATASET_VERSION}" \
  --validate-pool-only
test -n "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" || {
  echo "HF_TOKEN missing after sourcing hf-session.env; fail-closed eval cannot launch" >&2
  exit 2
}
if [[ "${WANDB_MODE}" == "online" ]]; then
  test -n "${WANDB_API_KEY:-}" || { echo "WANDB_API_KEY missing after sourcing wandb-session.env" >&2; exit 2; }
  if ! "${TRAIN_VENV}/bin/python" -c "import wandb" >/dev/null 2>&1; then
    echo "wandb missing from TRAIN_VENV; production checkpoint durability cannot proceed" >&2
    exit 2
  fi
fi
echo "RESUME_MODE=${RESUME_MODE} DATASET=${DATASET_ID}/${DATASET_VERSION} WANDB_PROJECT=${WANDB_PROJECT} WANDB_MODE=${WANDB_MODE} WANDB_RUN_NAME=${WANDB_RUN_NAME} ALLOW_LOCAL_ONLY=${ALLOW_LOCAL_ONLY}"
bash "${RUN_DIR}/launch_arm.sh"
EOF
chmod +x "${RUN_DIR}/train_one.sh"

JOB=$(sbatch --parsable \
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
  --job-name=skillit-370m \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/train-%A_%a.out" \
  --error="${RUN_DIR}/logs/train-%A_%a.err" \
  --export=ALL,RUN_DIR="${RUN_DIR}" \
  --wrap="bash ${RUN_DIR}/train_one.sh")

echo "job_id=${JOB}"
echo "RUN_DIR=${RUN_DIR}"
echo "POOL_DIR=${POOL_DIR}"
echo "DATASET_ID=${DATASET_ID}"
echo "DATASET_VERSION=${DATASET_VERSION}"
echo "RESUME_MODE=${RESUME_MODE}"
echo "TRAIN_VENV=${TRAIN_VENV}"
echo "SAVE_ROOT=${SAVE_ROOT}"
echo "PROGRESS_ROOT=${PROGRESS_ROOT}"
echo "WANDB_PROJECT=${WANDB_PROJECT} WANDB_MODE=${WANDB_MODE} WANDB_GROUP=${WANDB_GROUP}"
echo "arms (${N_ARMS}): ${ARM_IDS[*]}"
echo "${JOB}" > "${RUN_DIR}/job_id.txt"
