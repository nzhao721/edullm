#!/usr/bin/env bash
# RunPod 8×A100: MixLaw 370M validation control arm (mix01, 10B tokens).
# Code comes from a local tarball staged to S3 (no GitHub clone).
set -Eeuo pipefail

RUN_DIR="${RUN_DIR:-/workspace/mixlaw-370m-mix01}"
REPO_DIR="${REPO_DIR:-/workspace/edullm}"
MIX_NAME="${MIX_NAME:-mix01}"
NPROC="${NPROC:-8}"
AWS_ENV="${AWS_ENV:-/workspace/aws-session.env}"
WANDB_ENV="${WANDB_ENV:-/workspace/wandb-session.env}"
HF_ENV="${HF_ENV:-/workspace/hf-session.env}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
LADDER_BASE_CONFIG="${LADDER_BASE_CONFIG:-/workspace/ladder-base-config.yaml}"
LADDER_BASE_S3="${LADDER_BASE_S3:-s3://edullm-checkpoints/olmo2-370m-cpt/edullm-370M-30B/step5000-unsharded/config.yaml}"
CODE_S3_URI="${CODE_S3_URI:-s3://edullm-checkpoints/runpod/mixlaw-local-code.tgz}"
RECOVERY_MODE="${RECOVERY_MODE:-fail}"
RESUME_LOAD_PATH="${RESUME_LOAD_PATH:-}"

export OLMO_FLASH_ATTENTION=1
export OLMO_ATTN_BACKEND=flash_2
export OLMO_FUSED_LOSS=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TASK_LOSS_NPROC="${TASK_LOSS_NPROC:-${NPROC}}"
export LADDER_BASE_CONFIG

mkdir -p "${RUN_DIR}"/{logs,save,progress,pool,work}

if [[ -f "${AWS_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${AWS_ENV}"
fi
if [[ -f "${WANDB_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${WANDB_ENV}"
fi
if [[ -f "${HF_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${HF_ENV}"
fi
# Ensure torchrun workers see HF auth even if a child launcher re-sources env.
if [[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN}}"
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
  mkdir -p "${HOME}/.cache/huggingface"
  printf '%s' "${HF_TOKEN}" > "${HOME}/.cache/huggingface/token"
  # Also stage next to RUN_DIR for launch_validation lookup.
  cp -f "${HF_ENV}" "${RUN_DIR}/hf-session.env" 2>/dev/null || true
  echo "[hf] token installed for Hub downloads"
else
  echo "[hf] WARNING: no HF_TOKEN in environment"
fi

# Materialize code once at bootstrap. Recovery can reuse a patched local tree.
if [[ "${REUSE_LOCAL_CODE:-0}" == "1" ]]; then
  echo "[code] reusing local tree ${REPO_DIR}"
else
  rm -rf "${REPO_DIR}"
  mkdir -p "${REPO_DIR}"
  echo "[code] fetching ${CODE_S3_URI} → ${REPO_DIR}"
  aws s3 cp "${CODE_S3_URI}" /workspace/mixlaw_local_code.tgz
  tar -xzf /workspace/mixlaw_local_code.tgz -C "${REPO_DIR}"
fi
test -f "${REPO_DIR}/experiments/skill-dag/mixlaw/launch_validation_370m.sh"

PYTHON=python3
TORCH_VER="$(${PYTHON} -c 'import torch; print(torch.__version__)')"
echo "torch==${TORCH_VER}" > /tmp/pip-constraints.txt
${PYTHON} -m pip install -q -U pip wheel
# PyPI ai2-olmo==0.6.0 lacks OLMES *_val/test_rc_5shot_bpb labels needed for
# ladder task-loss → W&B eval/* metrics. Install from GitHub with [train] extras
# (datasets, torchmetrics, scikit-learn, …) so eval imports succeed.
${PYTHON} -m pip install -q -c /tmp/pip-constraints.txt \
  "ai2-olmo-core>=2.0" \
  "ai2-olmo[train] @ git+https://github.com/allenai/OLMo.git" \
  "edullm-data @ git+https://github.com/edu-llm/edullm-data@main" \
  boto3 wandb pyyaml tqdm awscli
${PYTHON} -m pip install -q --no-cache-dir \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
if [[ -n "${CODE_WANDB_ARTIFACT:-}" ]]; then
  echo "[code] applying ${CODE_WANDB_ARTIFACT} from W&B"
  rm -rf /workspace/mixlaw_wandb_code
  CODE_WANDB_ARTIFACT="${CODE_WANDB_ARTIFACT}" ${PYTHON} - <<'PY'
import os
import wandb

wandb.Api().artifact(os.environ["CODE_WANDB_ARTIFACT"]).download(
    root="/workspace/mixlaw_wandb_code"
)
PY
  cp -a /workspace/mixlaw_wandb_code/. "${REPO_DIR}/"
fi
${PYTHON} -c "import flash_attn; print('flash_attn ok', flash_attn.__version__)"
${PYTHON} -c "import datasets, torchmetrics, sklearn; import olmo; from olmo.eval.downstream import label_to_task_map; assert 'arc_easy_test_rc_5shot_bpb' in label_to_task_map, sorted(k for k in label_to_task_map if 'arc_easy' in k); print('ai2-olmo ok', 'arc_easy_test_rc_5shot_bpb present')"

if [[ ! -f "${LADDER_BASE_CONFIG}" ]]; then
  echo "[eval] fetching ladder base config from ${LADDER_BASE_S3}"
  aws s3 cp "${LADDER_BASE_S3}" "${LADDER_BASE_CONFIG}"
fi

export PYTHONPATH="${REPO_DIR}/experiments/skill-dag/mixlaw:${REPO_DIR}/experiments/token-selection:${REPO_DIR}/experiments/curriculum:${PYTHONPATH:-}"

nvidia-smi -L
for i in $(seq 1 36); do
  if ${PYTHON} -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    break
  fi
  echo "waiting for cuda ($i/36)..."
  sleep 5
done
${PYTHON} -c "import olmo_core, torch; print('olmo_core ok', 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"

MIXLAW_DIR="${REPO_DIR}/experiments/skill-dag/mixlaw"
RECIPE="${MIXLAW_DIR}/validation_mixtures_10b.json"

echo "[stage] preparing ${MIX_NAME} sidecars + peak pool from edullm-data..."
${PYTHON} "${MIXLAW_DIR}/prepare_validation_370m_data.py" \
  --recipe "${RECIPE}" \
  --work "${RUN_DIR}/work" \
  --only "${MIX_NAME}" \
  --stage-pool "${RUN_DIR}/pool" \
  2>&1 | tee "${RUN_DIR}/logs/stage.log"

export EDULLM_ROOT="${REPO_DIR}"
export MIX_NAME
export NPROC
export MIX_WEIGHTS_JSON="${RUN_DIR}/work/${MIX_NAME}/mix_weights.json"
export SAVE_FOLDER="${RUN_DIR}/save/checkpoints"
export PROGRESS_DIR="${RUN_DIR}/progress"
export POOL_DIR="${RUN_DIR}/pool"
export RUN_DIR
export LENGTH_TOKENS=10000000000
# Checkpoints live on runtime scratch + W&B; S3 is only for bootstrap data/code.
export S3_EXPORT=0
export WANDB_PROJECT="${WANDB_PROJECT:-mixlaw}"
export WANDB_GROUP="${WANDB_GROUP:-370m-validation}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-mixlaw-370m-${MIX_NAME}-runpod-$(date -u +%Y%m%d-%H%M%S)}"
export RECOVERY_MODE

# Resolve recovery only after an operator explicitly selects resume. The active
# volume is never scanned and an old checkpoint is never silently selected.
export DURABLE_METADATA_PATH="${DURABLE_METADATA_PATH:-${PROGRESS_DIR}/last_durable_step.json}"
case "${RECOVERY_MODE}" in
  fresh)
    rm -f "${PROGRESS_DIR}/wandb_run_id.txt" 2>/dev/null || true
    ;;
  resume)
    if [[ -n "${RESUME_LOAD_PATH}" ]]; then
      EXTRA_ARGS="--load-path ${RESUME_LOAD_PATH}"
    elif [[ -f "${DURABLE_METADATA_PATH}" ]]; then
      echo "[recovery] explicit resume: local durable metadata ${DURABLE_METADATA_PATH}"
    else
      echo "[recovery] RECOVERY_MODE=resume requires RESUME_LOAD_PATH or ${DURABLE_METADATA_PATH}" >&2
      exit 2
    fi
    ;;
  fail)
    ;;
  *)
    echo "RECOVERY_MODE must be fresh|resume|fail (got ${RECOVERY_MODE})" >&2
    exit 2
    ;;
esac
export EXTRA_ARGS="${EXTRA_ARGS:-} --no-auto-stage --device-batch-size ${DEVICE_BATCH_SIZE} --wandb-run-name ${WANDB_RUN_NAME}"

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY missing; set WANDB_MODE=disabled or provide wandb-session.env" >&2
  exit 2
fi

echo "[train] MIX_NAME=${MIX_NAME} NPROC=${NPROC} TASK_LOSS_NPROC=${TASK_LOSS_NPROC} RECOVERY_MODE=${RECOVERY_MODE} WANDB_MODE=${WANDB_MODE} DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE}"
bash "${MIXLAW_DIR}/launch_validation_370m.sh" \
  2>&1 | tee "${RUN_DIR}/logs/train.log"

echo "MIXLAW_DONE log=${RUN_DIR}/logs/train.log"
tail -n 40 "${RUN_DIR}/logs/train.log" || true
sleep 7200
