#!/usr/bin/env bash
# Quick 2-GPU gather test for CE + BLADE step2384 legacy state.pt checkpoints.
set -Eeuo pipefail

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/users/nzhao2/agent-runs}"
REPO_SCRIPTS="${REPO_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RUN_DIR="${RUN_DIR:-$SCRATCH_BASE/gather-hsdp-test-$(date -u +%Y%m%dT%H%M%SZ)}"

CE_CKPT="${CE_CKPT:-$SCRATCH_BASE/ce-regmix10b-models-20260727T212156Z/checkpoints/step2384}"
BLADE_CKPT="${BLADE_CKPT:-$SCRATCH_BASE/blade-regmix10b-models-20260727T212156Z/checkpoints/step2384}"
CE_OUT="${CE_OUT:-$SCRATCH_BASE/ce-regmix10b-models-20260727T212156Z/models/step2384/model.pt}"
BLADE_OUT="${BLADE_OUT:-$SCRATCH_BASE/blade-regmix10b-models-20260727T212156Z/models/step2384/model.pt}"

mkdir -p "$RUN_DIR/logs"
cp -f "$REPO_SCRIPTS/gather_hsdp_state_pt_to_model.py" "$RUN_DIR/"
cp -f "$REPO_SCRIPTS/gather_hsdp_state_pt.sbatch" "$RUN_DIR/"
chmod +x "$RUN_DIR/gather_hsdp_state_pt.sbatch"

submit_one() {
  local label=$1 ckpt=$2 out=$3
  cd "$RUN_DIR"
  export RUN_DIR CHECKPOINT_DIR="$ckpt" OUT_PT="$out"
  local job
  job=$(sbatch --exclude=wheat-01 \
    --job-name="gather-${label}" \
    --export=ALL,RUN_DIR,CHECKPOINT_DIR,OUT_PT \
    "$RUN_DIR/gather_hsdp_state_pt.sbatch" | awk '{print $NF}')
  echo "SUBMITTED $label job=$job ckpt=$ckpt out=$out"
  echo "$job" > "$RUN_DIR/job_${label}.txt"
}

submit_one ce-regmix10b "$CE_CKPT" "$CE_OUT"
submit_one blade-regmix10b "$BLADE_CKPT" "$BLADE_OUT"
squeue -u "$(whoami)"
