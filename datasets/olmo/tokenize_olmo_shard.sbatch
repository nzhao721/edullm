#!/usr/bin/env bash
#SBATCH --job-name=olmo-tok
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/tokenize_%A_%a.out
#SBATCH --error=logs/tokenize_%A_%a.err
#SBATCH --exclude=wheat-01
set -euo pipefail

source "${RUN_DIR}/env.sh"
source "${RUN_DIR}/venv/bin/activate"
export HF_HOME="${RUN_DIR}/hf-cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"
export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

MAP_FILE="${RUN_DIR}/tokenize_map.txt"
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MAP_FILE}")"
if [[ -z "${LINE}" ]]; then
  echo "No map line for task ${SLURM_ARRAY_TASK_ID}"
  exit 0
fi
IN_PATH="${LINE%%|*}"
OUT_PATH="${LINE#*|}"

if [[ -f "${OUT_PATH}" && -f "${OUT_PATH%.npy}.json" ]]; then
  echo "Skip existing ${OUT_PATH}"
  exit 0
fi

echo "[$(date -Is)] tokenize ${IN_PATH} -> ${OUT_PATH}"
python "${RUN_DIR}/scripts/tokenize_olmo_shard.py" \
  --input "${IN_PATH}" \
  --output "${OUT_PATH}" \
  --tokenizer allenai/dolma2-tokenizer \
  --eos-token-id 100257 \
  --hf-token-file "${HF_TOKEN_FILE}"
echo "[$(date -Is)] done"
