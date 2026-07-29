#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=regmix-lm-prep
#SBATCH --output=logs/lm-prep-%j.out
#SBATCH --error=logs/lm-prep-%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${SHARD_MANIFEST:?}" "${LM_ROOT:?}" "${REFHQ_ROOT:?}"

source "${VENV}/bin/activate"
export PYTHONPATH="${RUN_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${RUN_DIR}/logs" "${LM_ROOT}/checkpoints"

python "${RUN_DIR}/scripts/build_regmix_lm_chunks.py" \
  --shard-manifest "${SHARD_MANIFEST}" \
  --out-root "${LM_ROOT}" \
  --target-tokens-per-chunk "${TARGET_TOKENS_PER_CHUNK:-25000000}"

python "${RUN_DIR}/scripts/average_refhq_checkpoints.py" \
  --checkpoint "${REFHQ_ROOT}/step1000/model_eval.pt" \
  --checkpoint "${REFHQ_ROOT}/step1125/model_eval.pt" \
  --checkpoint "${REFHQ_ROOT}/step1315/model_eval.pt" \
  --out "${LM_ROOT}/checkpoints/refhq-step1000-1125-1315-avg.model_eval.pt"
