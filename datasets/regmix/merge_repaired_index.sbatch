#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=regmix-merge-ntok
#SBATCH --output=logs/merge_repaired_index_%j.out
#SBATCH --error=logs/merge_repaired_index_%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${REGMIX_ROOT:?}"

# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"

python "${RUN_DIR}/scripts/rebuild_stream_n_tokens.py" \
  --regmix-root "${REGMIX_ROOT}" \
  --skip-retokenize \
  --lengths-dir "${RUN_DIR}/stream_lengths" \
  --out-index "${RUN_DIR}/lm_labels_stream_aligned/metrics_index.jsonl.gz"
