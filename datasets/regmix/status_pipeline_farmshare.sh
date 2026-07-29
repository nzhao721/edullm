#!/usr/bin/env bash
SOCKET=/tmp/farmshare-nzhao2.sock
RUN=/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810
ssh -S "$SOCKET" -o BatchMode=yes nzhao2@login.farmshare.stanford.edu bash <<'REMOTE'
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
RUN=/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810
echo "=== controller ==="
squeue -u nzhao2 -n regmix-lm-pipe -o "%.18i %.2t %.10M %R" 2>/dev/null || true
echo "=== gpu chunks running ==="
squeue -u nzhao2 -t RUNNING -p gpu -o "%.18i %.9j %.10M %R" 2>/dev/null | grep nzhao2 || true
echo "=== progress ==="
bash /scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810/scripts/check_lm_progress.sh 2>/dev/null | head -15
echo "=== pipeline log ==="
ls -t "$RUN"/logs/lm-pipe-ctrl-*.out 2>/dev/null | head -1 | xargs tail -8 2>/dev/null || echo no pipeline log
REMOTE
