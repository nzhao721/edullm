#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=relema-unshard-smoke
#SBATCH --output=logs/relema_unshard_smoke_%j.out
#SBATCH --error=logs/relema_unshard_smoke_%j.err

set -Eeuo pipefail
: "${RUN_DIR:?}"

cd "$RUN_DIR"
mkdir -p logs

LADDER_VENV="${LADDER_VENV:-/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/venv}"
# shellcheck disable=SC1091
source "$LADDER_VENV/bin/activate"

export OUT_PT="${OUT_PT:-$RUN_DIR/relema_step2386_model.pt}"
bash "$RUN_DIR/unshard_relema_step2386.sh"

SMOKE_OUT="${SMOKE_OUT:-$RUN_DIR/logs/smoke_relema_cpu_results.json}"
python "$RUN_DIR/smoke_test_inference_370m.py" \
  --device cpu \
  --only-relema \
  --relema "$OUT_PT" \
  --out "$SMOKE_OUT"

echo "DONE unshard+smoke -> $OUT_PT ; results $SMOKE_OUT"
