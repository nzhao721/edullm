#!/usr/bin/env bash
# Thin matrix launcher over the 17 curriculum arms.
#
# Does NOT submit AWS jobs by itself. It either:
#   * prints the arm matrix (default / --print-only), or
#   * invokes launch_arm.sh once per arm when SUBMIT_CMD is set, e.g.:
#       SUBMIT_CMD='echo would-submit' bash submit_matrix.sh
#       SUBMIT_CMD='sbatch my_template.sbatch' bash submit_matrix.sh   # caller owns the template
#
# Shared env expected by launch_arm.sh (plus per-arm overrides below):
#   SAVE_ROOT           parent of per-arm dirs; SAVE_FOLDER=$SAVE_ROOT/<arm_id>/checkpoints
#   PROGRESS_ROOT       parent of per-arm dirs; PROGRESS_DIR=$PROGRESS_ROOT/<arm_id>/progress
#   TRAIN_PATHS_FILE    control flat memmap list
#   CURRICULUM_INDEX    curriculum index root
#   NPROC, DEVICE_BATCH_SIZE, SEED, LR_ALPHA_F, EXTRA_ARGS  (optional)
#
# Local sync mirrors S3 layout:
#   s3://edullm-checkpoints/curriculum/<arm_id>/checkpoints
#   s3://edullm-checkpoints/curriculum/<arm_id>/progress
#
# Arm IDs (17):
#   control
#   linear10-cr   linear10-flesch   linear10-mtld   linear10-learn
#   expand-cr     expand-flesch     expand-mtld     expand-learn
#   warmup-cr     warmup-flesch     warmup-mtld     warmup-learn
#   interleave-cr interleave-flesch interleave-mtld interleave-learn
#
# Optional post-hoc EMA (after training, per arm; task-loss default on):
#   python experiments/curriculum/ema_merge_checkpoints.py \
#     --checkpoints-root "$SAVE_ROOT/<arm_id>/checkpoints" \
#     --arm-id <arm_id>
#   # skip eval: add --no-task-loss
#   # all 17: for arm in control linear10-cr ...; do ... --arm-id "$arm"; done

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
To drive launches, set SAVE_ROOT / PROGRESS_ROOT / data paths and e.g.:

  export SAVE_ROOT=/scratch/curriculum
  export PROGRESS_ROOT=/scratch/curriculum-progress
  export TRAIN_PATHS_FILE=/data/regmix/paths_train.txt
  export CURRICULUM_INDEX=/data/regmix/curriculum
  export NPROC=1
  export SUBMIT_CMD='bash'   # or an AWS/slurm wrapper that runs the given command
  bash experiments/curriculum/launch/submit_matrix.sh

EOF
  exit 0
fi

: "${SAVE_ROOT:?SAVE_ROOT is required when SUBMIT_CMD is set}"
: "${PROGRESS_ROOT:?PROGRESS_ROOT is required when SUBMIT_CMD is set}"

for entry in "${ARMS[@]}"; do
  IFS='|' read -r ARM_ID PACING METRIC <<<"${entry}"
  export ARM_ID PACING
  export SAVE_FOLDER="${SAVE_ROOT}/${ARM_ID}/checkpoints"
  export PROGRESS_DIR="${PROGRESS_ROOT}/${ARM_ID}/progress"
  if [[ "${PACING}" == "control" ]]; then
    unset DIFFICULTY_METRIC || true
    : "${TRAIN_PATHS_FILE:?TRAIN_PATHS_FILE required}"
  else
    export DIFFICULTY_METRIC="${METRIC}"
    : "${CURRICULUM_INDEX:?CURRICULUM_INDEX required}"
  fi
  echo "[submit_matrix] ${SUBMIT_CMD} ${LAUNCH}  # ${ARM_ID}"
  # SUBMIT_CMD receives the launch script path; wrappers may sbatch/torchrun around it.
  # shellcheck disable=SC2086
  ${SUBMIT_CMD} "${LAUNCH}"
done
