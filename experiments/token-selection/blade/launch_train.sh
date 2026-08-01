#!/usr/bin/env bash
# Hardware-agnostic BLADE launch for ephemeral machines.
# World size comes from torchrun / env — never hardcode GPUs.
#
# Scratch starts empty: stage train+RefHQ from s3://edullm-data (prepare_blade_data.py).
# Artifacts remain on scratch and upload to W&B; production online uploads fail closed.
#
# W&B project token-selection is the artifact store. Push
# wandb-session.env via scripts/farmshare/push_wandb_session_to_farmshare.sh
# "$RUN_DIR" (or set WANDB_SESSION_ENV). Local smoke: WANDB_MODE=disabled.
# Cross-job resume: WANDB_RESUME_ARTIFACT=entity/project/name:alias.
# Do not assume FarmShare/laptop corpora, old run dirs, or legacy edullm-datasets.
#
# Required:
#   --name, --save-folder, --progress-dir
#   plus either BLADE_WORK (stages from edullm-data) or explicit path-list + length args
#
# Example (clean / ephemeral node):
#   BLADE_WORK=/scratch/$USER/blade-work \
#   bash experiments/token-selection/blade/launch_train.sh \
#     --name blade-regmix10b-v2 \
#     --save-folder /scratch/$USER/ckpts/blade-regmix10b-v2 \
#     --progress-dir /scratch/$USER/runs/blade-regmix10b-v2 \
#     --fresh
#
# Or pass path lists after an in-job prepare:
#   bash experiments/token-selection/blade/launch_train.sh \
#     --name blade-regmix10b-v2 \
#     --train-paths-file "$BLADE_WORK/train_tokenized/paths_train.txt" \
#     --ref-paths-file "$BLADE_WORK/ref_tokenized/paths_refhq.txt" \
#     --save-folder ... --progress-dir ... \
#     --length-tokens "$(cat "$BLADE_WORK/length_tokens.txt")" \
#     --fresh
#
# Optional env:
#   BLADE_WORK       stage root; when set and path args omitted, run prepare_blade_data.py
#   NPROC_PER_NODE   default: 1 (or count of CUDA_VISIBLE_DEVICES if set)
#   MASTER_ADDR / MASTER_PORT
#   TASK_LOSS_EVAL=0 to disable post-save eval spawn
#   TASK_LOSS_EVAL_SCRIPT / TASK_LOSS_DIR
#   WANDB_RESUME_ARTIFACT    restore a checkpoint artifact into scratch
#   LOAD_PATH                explicit local step dir
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${TS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  NPROC="${NPROC_PER_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # Count comma-separated device ids (empty → 0 → fall back to 1).
  IFS=',' read -r -a _devs <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC="${#_devs[@]}"
  [[ "${NPROC}" -ge 1 ]] || NPROC=1
else
  NPROC=1
fi

export TASK_LOSS_STRICT=1
export TASK_LOSS_NPROC="${TASK_LOSS_NPROC:-${NPROC}}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

ARGS=("$@")

has_flag() {
  local flag="$1"
  local a
  for a in "${ARGS[@]+"${ARGS[@]}"}"; do
    [[ "${a}" == "${flag}" || "${a}" == "${flag}="* ]] && return 0
  done
  return 1
}

# Require either BLADE_WORK (clean-machine stage) or explicit path lists.
if [[ -z "${BLADE_WORK:-}" ]]; then
  if ! has_flag "--train-paths-file" || ! has_flag "--ref-paths-file"; then
    echo "launch_train: set BLADE_WORK to stage from s3://edullm-data, or pass" >&2
    echo "  --train-paths-file and --ref-paths-file from an in-job prepare." >&2
    echo "  Do not rely on pre-existing scratch or legacy edullm-datasets paths." >&2
    exit 1
  fi
fi

# When BLADE_WORK is set and path lists are not already on the CLI, stage from edullm-data.
if [[ -n "${BLADE_WORK:-}" ]]; then
  if ! has_flag "--train-paths-file" || ! has_flag "--ref-paths-file"; then
    echo "launch_train: staging corpora into BLADE_WORK=${BLADE_WORK} from edullm-data"
    mkdir -p "${BLADE_WORK}"
    python3 "${SCRIPT_DIR}/prepare_blade_data.py" --work "${BLADE_WORK}"
  fi
  if ! has_flag "--train-paths-file"; then
    ARGS+=(--train-paths-file "${BLADE_WORK}/train_tokenized/paths_train.txt")
  fi
  if ! has_flag "--ref-paths-file"; then
    ARGS+=(--ref-paths-file "${BLADE_WORK}/ref_tokenized/paths_refhq.txt")
  fi
  if ! has_flag "--length-tokens"; then
    ARGS+=(--length-tokens "$(cat "${BLADE_WORK}/length_tokens.txt")")
  fi
  if [[ -f "${BLADE_WORK}/blade_data_summary.json" ]] && ! has_flag "--token-dtype"; then
    _dtype="$(
      BLADE_WORK="${BLADE_WORK}" python3 -c \
        'import json,os; p=os.environ["BLADE_WORK"]+"/blade_data_summary.json"; print(json.load(open(p))["train"].get("dtype","uint32"))'
    )"
    ARGS+=(--token-dtype "${_dtype}")
  fi
fi

if [[ -n "${TASK_LOSS_DIR:-}" ]] && ! has_flag "--task-loss-dir"; then
  ARGS+=(--task-loss-dir "${TASK_LOSS_DIR}")
fi
if [[ -n "${TASK_LOSS_EVAL_SCRIPT:-}" ]] && ! has_flag "--task-loss-eval-script"; then
  ARGS+=(--task-loss-eval-script "${TASK_LOSS_EVAL_SCRIPT}")
fi
if [[ "${TASK_LOSS_EVAL:-1}" == "0" ]] && ! has_flag "--no-task-loss-eval"; then
  ARGS+=(--no-task-loss-eval)
fi
if [[ -n "${LOAD_PATH:-}" ]] && ! has_flag "--load-path"; then
  ARGS+=(--load-path "${LOAD_PATH}")
fi
if [[ -n "${WANDB_RESUME_ARTIFACT:-}" ]] && ! has_flag "--wandb-resume-artifact"; then
  ARGS+=(--wandb-resume-artifact "${WANDB_RESUME_ARTIFACT}")
fi

_BLADE_RUN_NAME="${NAME:-}"
if [[ -z "${_BLADE_RUN_NAME}" ]]; then
  for ((i = 0; i < ${#ARGS[@]}; i++)); do
    a="${ARGS[$i]}"
    if [[ "${a}" == "--name" && $((i + 1)) -lt ${#ARGS[@]} ]]; then
      _BLADE_RUN_NAME="${ARGS[$((i + 1))]}"
      break
    elif [[ "${a}" == --name=* ]]; then
      _BLADE_RUN_NAME="${a#--name=}"
      break
    fi
  done
fi

echo "launch_train: nproc_per_node=${NPROC} master=${MASTER_ADDR}:${MASTER_PORT}"
# shellcheck disable=SC1091
source "${TS_ROOT}/token_selection/scripts/wandb_env.sh" "blade" "${_BLADE_RUN_NAME}"
exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/train_blade_olmo_370m.py" \
  "${ARGS[@]}"
