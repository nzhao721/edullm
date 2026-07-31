#!/usr/bin/env bash
# Launch one OLMo2-370M CE arm on a mixlaw 370M validation mixture.
#
# Domain weights come from the materialized corpus at
#   s3://edullm-datasets/mixlaw/mixes/<MIX_NAME>/
# (recipe: validation_mixtures_10b.json). This launcher does not re-sample
# weights — it trains on the fixed 10B tokenized slice for MIX_NAME.
#
# Required:
#   MIX_NAME           e.g. ML-pilot_caps | ML-near-opt-4 | LGB-near-opt-8 | mix07
#   TRAIN_PATHS_FILE   local paths_train.txt for that mix (after prepare/sync)
#   SAVE_FOLDER        checkpoint root
#   PROGRESS_DIR       metrics / run_meta
#
# Optional:
#   NPROC              ranks (default 1; >1 → torchrun)
#   LENGTH_TOKENS      default 10000000000
#   EXTRA_ARGS         forwarded to the control trainer
#
# Example:
#   MIX_NAME=ML-near-opt-4 \
#   TRAIN_PATHS_FILE=/data/mixlaw/ML-near-opt-4/paths_train.txt \
#   SAVE_FOLDER=/ckpt/ML-near-opt-4 PROGRESS_DIR=/prog/ML-near-opt-4 \
#   NPROC=4 bash launch_validation_370m.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONTROL_DIR="${REPO_ROOT}/experiments/token-selection/control"
export PYTHONPATH="${REPO_ROOT}/experiments/token-selection${PYTHONPATH:+:${PYTHONPATH}}"

: "${MIX_NAME:?Set MIX_NAME to a validation_mixtures_10b.json run_name}"
: "${TRAIN_PATHS_FILE:?Set TRAIN_PATHS_FILE to the mix paths_train.txt}"
: "${SAVE_FOLDER:?Set SAVE_FOLDER}"
: "${PROGRESS_DIR:?Set PROGRESS_DIR}"

NPROC="${NPROC:-1}"
LENGTH_TOKENS="${LENGTH_TOKENS:-10000000000}"
NAME="${NAME:-mixlaw-370m-${MIX_NAME}}"

TRAINER="${CONTROL_DIR}/train_ce_regmix_olmo_370m.py"
COMMON_ARGS=(
  --name "${NAME}"
  --train-paths-file "${TRAIN_PATHS_FILE}"
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --length-tokens "${LENGTH_TOKENS}"
)
# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS:-} )

echo "[launch_validation_370m] MIX_NAME=${MIX_NAME} NPROC=${NPROC} NAME=${NAME}" >&2

if [[ "${NPROC}" -eq 1 ]]; then
  exec python "${TRAINER}" "${COMMON_ARGS[@]}" "${EXTRA[@]}"
fi

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  "${TRAINER}" \
  "${COMMON_ARGS[@]}" \
  "${EXTRA[@]}"
