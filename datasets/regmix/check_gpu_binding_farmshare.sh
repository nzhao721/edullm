#!/usr/bin/env bash
SOCKET=/tmp/farmshare-nzhao2.sock
ssh -S "$SOCKET" -o BatchMode=yes nzhao2@login.farmshare.stanford.edu bash <<'REMOTE'
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
echo "=== running regmix chunk jobs: node + gres ==="
squeue -u nzhao2 -n regmix-lm-chunk -t RUNNING -o "%.18i %N %b %R" 2>/dev/null
echo "=== nodes with multiple nzhao2 chunk jobs ==="
squeue -u nzhao2 -n regmix-lm-chunk -t RUNNING -h -o "%N" 2>/dev/null | sort | uniq -c | awk '$1>1'
echo "=== compare: historical retry jobs node packing (sample) ==="
sacct -u nzhao2 --name=regmix-lm-retry --state=COMPLETED -S 2026-07-28 --format=JobID,NodeList,AllocTRES%40 -P -n 2>/dev/null | tail -10
echo "=== env from running chunk job (if any) ==="
jid=$(squeue -u nzhao2 -n regmix-lm-chunk -t RUNNING -h -o "%i" 2>/dev/null | head -1)
if [ -n "$jid" ]; then
  srun --jobid="$jid" --overlap env 2>/dev/null | grep -E 'CUDA|SLURM.*GPU|CHUNK' || echo "srun env failed for $jid"
fi
REMOTE
