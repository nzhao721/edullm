#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Retokenize one domain; batch HF encode is mostly single-thread + GIL.
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=regmix-stream-ntok
#SBATCH --output=logs/rebuild_stream_n_tokens_%j.out
#SBATCH --error=logs/rebuild_stream_n_tokens_%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${REGMIX_ROOT:?}" "${SCRIPTS:?}" "${DOMAIN:?}"

# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"

export HF_HOME="${HF_HOME:-${REGMIX_ROOT}/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/stream_lengths" "${RUN_DIR}/lm_labels_stream_aligned"

python "${SCRIPTS}/rebuild_stream_n_tokens.py" \
  --regmix-root "${REGMIX_ROOT}" \
  --domains "${DOMAIN}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --lengths-dir "${RUN_DIR}/stream_lengths" \
  --lengths-only
