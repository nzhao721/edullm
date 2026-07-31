#!/bin/bash
#SBATCH --job-name=slice-750m-fwedu
#SBATCH --partition=normal
#SBATCH --exclude=wheat-01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/nzhao2/agent-runs/fineweb-edu-750m-smollm2-tokenized/logs/slice-%j.out
#SBATCH --error=/scratch/users/nzhao2/agent-runs/fineweb-edu-750m-smollm2-tokenized/logs/slice-%j.er

set -euo pipefail

SRC="${SRC:-/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-tokenized}"
DST="${DST:-/scratch/users/nzhao2/agent-runs/fineweb-edu-750m-smollm2-tokenized}"
STAGING="${STAGING:-/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging/scripts/farmshare}"
VENV="${VENV:-/scratch/users/nzhao2/agent-runs/venvs/fineweb-tokenize}"
MAX_TOKENS="${MAX_TOKENS:-750000000}"

mkdir -p "${DST}/logs"
source "${VENV}/bin/activate"
# Fresh destination (do not overwrite 500M)
rm -rf "${DST}/train_tokens.bin" "${DST}/train_doc_ids.bin" "${DST}/train_positions.bin" "${DST}/meta.json"

python -u "${STAGING}/slice_tokenized_subset.py" \
  --src-dir "${SRC}" \
  --dst-dir "${DST}" \
  --max-tokens "${MAX_TOKENS}"

echo "done: $(date -Is)"
cat "${DST}/meta.json"
