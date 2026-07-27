#!/usr/bin/env bash
#SBATCH --job-name=flash-attn-install
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=wheat-01
#SBATCH --output=/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/logs/flash_attn_install_%j.out
#SBATCH --error=/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/logs/flash_attn_install_%j.err
set -euo pipefail
RUN_DIR=/scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217
source "${RUN_DIR}/env.sh"
source "${RUN_DIR}/venv/bin/activate"
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load cuda/12.9.0 || module load cuda/12.4.0 || true
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
python -m pip install -U ninja packaging
python -m pip install flash-attn==2.5.9.post1 --no-build-isolation || \
  python -m pip install flash-attn --no-build-isolation
python - <<'PY'
import flash_attn
print("flash_attn OK", getattr(flash_attn, "__version__", "?"))
PY
# Prefer flash for subsequent training jobs
sed -i '/OLMO_FLASH_ATTENTION/d' "${RUN_DIR}/env.sh"
echo 'export OLMO_FLASH_ATTENTION=1' >> "${RUN_DIR}/env.sh"
grep FLASH "${RUN_DIR}/env.sh"
