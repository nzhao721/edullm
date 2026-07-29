#!/usr/bin/env bash
SOCKET=/tmp/farmshare-nzhao2.sock
ssh -S "$SOCKET" -o BatchMode=yes nzhao2@login.farmshare.stanford.edu bash <<'REMOTE'
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
for jid in 1667563 1667572; do
  echo "=== job $jid ==="
  squeue -j "$jid" -o "%.18i %N %b" 2>/dev/null
  srun --jobid="$jid" --overlap bash -lc 'echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES SLURM_JOB_GPUS=$SLURM_JOB_GPUS SLURM_STEP_GPUS=$SLURM_STEP_GPUS nvidia-smi -L 2>/dev/null | head -2' 2>/dev/null || echo failed
done
echo "=== successful retry log sample (chunk index + batch) ==="
f=$(ls -t /scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810/logs/lm-retry-*_*.out 2>/dev/null | head -1)
echo file=$f
grep -E 'labeled_lm_chunk|batch_tokens|index' "$f" 2>/dev/null | tail -3
echo "=== failed chunk log sample ==="
f2=$(ls -t /scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810/logs/lm-chunk-*_*.err 2>/dev/null | head -1)
echo file=$f2
grep -E 'labeled_lm_chunk|batch_tokens|CHUNK|index|OOM' "$f2" 2>/dev/null | tail -5
REMOTE
