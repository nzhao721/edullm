#!/usr/bin/env bash
set -Eeuo pipefail
ssh -S /tmp/farmshare-nzhao2.sock -o BatchMode=yes nzhao2@login.farmshare.stanford.edu bash -s <<'REMOTE'
set -e
TOP=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
MIX=/scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236
echo "=== topup download states ==="
sacct -j 1665880 --format=State -n | sort | uniq -c
echo "=== topup deps ==="
squeue --me -j 1666207,1666208,1666209,1665880 -o '%.12i %.10T %.20j' 2>/dev/null || true
echo "=== mixlaw ==="
squeue --me -j 1666297,1666298,1666299 -o '%.12i %.10T %.20j' 2>/dev/null || true
echo "=== mixlaw pool log ==="
tail -n 40 "$MIX/logs/pool-1666297.out" 2>/dev/null || echo 'no log'
echo "=== topup dl sample log ==="
ls "$TOP/logs"/dl-1665880_*.out 2>/dev/null | tail -3
tail -n 5 $(ls "$TOP/logs"/dl-1665880_*.out 2>/dev/null | tail -1) 2>/dev/null || true
REMOTE
