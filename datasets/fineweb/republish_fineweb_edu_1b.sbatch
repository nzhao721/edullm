#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=fw1b-repub
#SBATCH --output=logs/fineweb_edu_1b_repub_%j.out
#SBATCH --error=logs/fineweb_edu_1b_repub_%j.err

set -Eeuo pipefail
: "${RUN_DIR:?}" "${STAGE_DIR:?}" "${AWS_SESSION_ENV:?}" "${EDULLM_ROOT:?}"

export TOKENS_ONLY_STAGE="${RUN_DIR}/publish-stage-tokens"
export TOKENIZED_ROOT="${TOKENIZED_ROOT:-/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-tokenized}"
export RAW_ROOT="${RAW_ROOT:-/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-raw}"
export STAGE_DIR RUN_DIR AWS_SESSION_ENV

bash "${EDULLM_ROOT}/datasets/fineweb/republish_fineweb_edu_1b_tokens_and_raw.sh"
