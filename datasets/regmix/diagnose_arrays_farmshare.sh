#!/usr/bin/env bash
SOCKET=/tmp/farmshare-nzhao2.sock
ssh -S "$SOCKET" -o BatchMode=yes nzhao2@login.farmshare.stanford.edu bash <<'REMOTE'
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
RUN=/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810
ctrl=$(squeue -u nzhao2 -h -n regmix-lm-pipe -o "%i" 2>/dev/null | head -1)
echo "controller=$ctrl"
squeue -u nzhao2 | grep regmix
echo "=== progress ==="
bash "$RUN/scripts/check_lm_progress.sh" | head -12
echo "=== pipe log tail ==="
ls -t "$RUN/logs/lm-pipe-ctrl-"*.out 2>/dev/null | head -1 | xargs tail -8
echo "=== recent retry outcomes ==="
sacct -u nzhao2 -S "$(date -d '10 minutes ago' +%Y-%m-%dT%H:%M)" --name=regmix-lm-retry --format=State -P -n 2>/dev/null | sort | uniq -c
echo "=== latest retry err ==="
ls -t "$RUN/logs/lm-retry-"*.err 2>/dev/null | head -1 | xargs tail -8
REMOTE
