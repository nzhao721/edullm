#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<'EOF'
set -e
RUN=/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z
echo "=== logs dir ==="
ls -la "$RUN/logs/" | tail -20
echo "=== sbatch mem ==="
grep -E 'mem|cpus|time' "$RUN/scripts/olmohq/publish_resume.sbatch" "$RUN/scripts/olmohq/"*.sbatch 2>/dev/null | head -20
echo "=== slurm accounting detail ==="
sacct -j 1670701 --format=JobID,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,ReqMem,AllocTRES%40 -P
echo "=== dmesg/oom clues in slurmd? (often empty) ==="
# Check if out file has anything after 'before publish'
OUT="$RUN/logs/olmo127b_edullm_publish_1670701.out"
wc -l "$OUT"
tail -c 3000 "$OUT" | tr '\r' '\n' | tail -40
# Look for .e files with any suffix
find "$RUN/logs" -name '*1670701*' -ls
EOF
