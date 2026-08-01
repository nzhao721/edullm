#!/usr/bin/env bash
# Thin matrix launcher over the 17 curriculum arms.
#
# Does NOT submit AWS jobs by itself. It either:
#   * prints the arm matrix (default / --print-only), or
#   * invokes launch_arm.sh once per arm when SUBMIT_CMD is set, e.g.:
#       SUBMIT_CMD='echo would-submit' bash submit_matrix.sh
#       SUBMIT_CMD='sbatch my_template.sbatch' bash submit_matrix.sh   # caller owns the template
#
# Ephemeral runtime: set a fresh job-scoped RUN_DIR (or SAVE_ROOT/PROGRESS_ROOT
# under it). Scratch starts empty and is wiped after the job. Data stages from
# s3://edullm-data/. Checkpoints and all other run artifacts stay on scratch and
# upload to W&B project "curriculum" (push wandb-session.env to RUN_DIR).
# Production checkpoint uploads are synchronous and fail-closed.
#
# Shared env expected by launch_arm.sh (plus per-arm overrides below):
#   SAVE_ROOT           parent of per-arm dirs; SAVE_FOLDER=$SAVE_ROOT/<arm_id>/checkpoints
#   PROGRESS_ROOT       parent of per-arm dirs; PROGRESS_DIR=$PROGRESS_ROOT/<arm_id>/progress
#                       (metrics default to $PROGRESS_ROOT/<arm_id>/metrics)
#   RUN_DIR             job scratch root with aws-session.env + wandb-session.env
#   WANDB_PROJECT       default curriculum; WANDB_MODE default online
#   TRAIN_DATASET_ID    default pretrain/regmix-10b (edullm-data)
#   DATA_CACHE_DIR / EDULLM_DATA_CACHE  job-scoped staging root (fetch-if-missing)
#   TRAIN_PATHS_FILE / CURRICULUM_INDEX  optional THIS-job staged overrides
#   NPROC, DEVICE_BATCH_SIZE, SEED, LR_ALPHA_F, EXTRA_ARGS, FRESH, LOAD_PATH  (optional)
#
# Arm IDs (17):
#   control
#   linear10-cr   linear10-flesch   linear10-mtld   linear10-learn
#   expand-cr     expand-flesch     expand-mtld     expand-learn
#   warmup-cr     warmup-flesch     warmup-mtld     warmup-learn
#   interleave-cr interleave-flesch interleave-mtld interleave-learn
#
# Optional post-hoc EMA (using scratch checkpoints or W&B artifact downloads):
#   python experiments/curriculum/ema_merge_checkpoints.py \
#     --checkpoints-root "$SAVE_ROOT/<arm_id>/checkpoints" \
#     --arm-id <arm_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH="${SCRIPT_DIR}/launch_arm.sh"
PRINT_ONLY=0
if [[ "${1:-}" == "--print-only" ]]; then
  PRINT_ONLY=1
fi

# arm_id | pacing | difficulty_metric (empty for control)
ARMS=(
  "control|control|"
  "linear10-cr|linear_n10|compression_ratio"
  "linear10-flesch|linear_n10|flesch"
  "linear10-mtld|linear_n10|mtld"
  "linear10-learn|linear_n10|learnability"
  "expand-cr|expanding_25_1000|compression_ratio"
  "expand-flesch|expanding_25_1000|flesch"
  "expand-mtld|expanding_25_1000|mtld"
  "expand-learn|expanding_25_1000|learnability"
  "warmup-cr|warmup_1000|compression_ratio"
  "warmup-flesch|warmup_1000|flesch"
  "warmup-mtld|warmup_1000|mtld"
  "warmup-learn|warmup_1000|learnability"
  "interleave-cr|interleave_i10_linear|compression_ratio"
  "interleave-flesch|interleave_i10_linear|flesch"
  "interleave-mtld|interleave_i10_linear|mtld"
  "interleave-learn|interleave_i10_linear|learnability"
)

echo "# Curriculum CL matrix — 17 arms"
echo "# Columns: arm_id pacing difficulty_metric"
for entry in "${ARMS[@]}"; do
  IFS='|' read -r ARM_ID PACING METRIC <<<"${entry}"
  echo "${ARM_ID}  ${PACING}  ${METRIC:-—}"
done

if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  exit 0
fi

if [[ -z "${SUBMIT_CMD:-}" ]]; then
  cat <<'EOF'

No SUBMIT_CMD set — matrix printed only.
To drive launches on an ephemeral job scratch (data from edullm-data):

  export RUN_DIR="${TMPDIR:-/tmp}/curriculum-job-$$"
  mkdir -p "$RUN_DIR"
  export SAVE_ROOT="$RUN_DIR/arms"
  export PROGRESS_ROOT="$RUN_DIR/progress"
  export DATA_CACHE_DIR="$RUN_DIR/edullm-data-cache"
  export TRAIN_DATASET_ID=pretrain/regmix-10b
  export NPROC=1
  # FarmShare: push aws-session.env + wandb-session.env into RUN_DIR first
  #   bash scripts/farmshare/push_aws_session_to_farmshare.sh "$RUN_DIR"
  #   bash scripts/farmshare/push_wandb_session_to_farmshare.sh "$RUN_DIR"
  export WANDB_PROJECT=curriculum
  export WANDB_MODE=online
  export SUBMIT_CMD='bash'   # or an AWS/slurm wrapper that runs the given command
  bash experiments/curriculum/launch/submit_matrix.sh

Artifacts: job scratch + W&B project curriculum. S3 is input staging only.

EOF
  exit 0
fi

: "${SAVE_ROOT:?SAVE_ROOT is required when SUBMIT_CMD is set (job-scoped)}"
: "${PROGRESS_ROOT:?PROGRESS_ROOT is required when SUBMIT_CMD is set (job-scoped)}"

for entry in "${ARMS[@]}"; do
  IFS='|' read -r ARM_ID PACING METRIC <<<"${entry}"
  export ARM_ID PACING
  export SAVE_FOLDER="${SAVE_ROOT}/${ARM_ID}/checkpoints"
  export PROGRESS_DIR="${PROGRESS_ROOT}/${ARM_ID}/progress"
  if [[ "${PACING}" == "control" ]]; then
    unset DIFFICULTY_METRIC || true
  else
    export DIFFICULTY_METRIC="${METRIC}"
  fi
  echo "[submit_matrix] ${SUBMIT_CMD} ${LAUNCH}  # ${ARM_ID}"
  # SUBMIT_CMD receives the launch script path; wrappers may sbatch/torchrun around it.
  # shellcheck disable=SC2086
  ${SUBMIT_CMD} "${LAUNCH}"
done
