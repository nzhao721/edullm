#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=20:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=skillit-train
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -Eeuo pipefail
: "${RUN_DIR:?}"
cd "${RUN_DIR}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${RUN_DIR}/scripts}"
# shellcheck disable=SC1091
source "${SCRIPTS_DIR}/common.sh"
bash "${SCRIPTS_DIR}/setup_venv_no_aws.sh"
bash "${SCRIPTS_DIR}/launch_no_aws.sh"
