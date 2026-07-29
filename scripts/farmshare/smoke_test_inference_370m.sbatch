#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=smoke-inf
#SBATCH --output=logs/smoke_inference_%j.out
#SBATCH --error=logs/smoke_inference_%j.err

set -Eeuo pipefail
: "${RUN_DIR:?}"

cd "$RUN_DIR"
mkdir -p logs

LADDER_VENV="${LADDER_VENV:-/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/venv}"
# shellcheck disable=SC1091
source "$LADDER_VENV/bin/activate"

REFHQ_PT="${REFHQ_PT:-/scratch/users/nzhao2/agent-runs/refhq-unshard-20260727/reference/refhq_step1315_model.pt}"
CE_PT="${CE_PT:-/scratch/users/nzhao2/agent-runs/ce-regmix10b-models-20260727T212156Z/models/step2384/model.pt}"
BLADE_PT="${BLADE_PT:-/scratch/users/nzhao2/agent-runs/blade-regmix10b-models-20260727T212156Z/models/step2384/model.pt}"
OUT="${OUT:-$RUN_DIR/logs/smoke_inference_results.json}"

python "$RUN_DIR/smoke_test_inference_370m.py" \
  --refhq "$REFHQ_PT" \
  --ce "$CE_PT" \
  --blade "$BLADE_PT" \
  --out "$OUT"
