#!/usr/bin/env bash
set -euo pipefail
echo "=== identity ==="
whoami; hostname; date -u
echo "=== scratch ==="
df -h /scratch/users/nzhao2 | tail -1
du -sh /scratch/users/nzhao2
echo "=== agent-runs (tail) ==="
ls /scratch/users/nzhao2/agent-runs | tail -40
echo "=== fineweb / hf cache ==="
ls -d /scratch/users/nzhao2/agent-runs/*fineweb* 2>/dev/null || true
du -sh /scratch/users/nzhao2/.cache/huggingface 2>/dev/null || true
find /scratch/users/nzhao2 -maxdepth 5 \( -name '*sample-100BT*' -o -name '*entries.db*' -o -name 'fineweb_with_fullwiki*' \) 2>/dev/null | head -40
echo "=== partitions ==="
sinfo -s
sinfo -o '%P %a %l %D %T %c %m' | head -40
echo "=== my queue ==="
squeue -u nzhao2 | head -30
echo "=== python / hf ==="
python3 --version
which python3
test -f "$HOME/.cache/huggingface/token" && echo has_home_hf_token || echo no_home_hf_token
test -f /scratch/users/nzhao2/.cache/huggingface/token && echo has_scratch_hf_token || echo no_scratch_hf_token
ls /scratch/users/nzhao2/agent-runs/venvs 2>/dev/null || true
echo "=== node cpu samples ==="
scontrol show node oat-01 2>/dev/null | tr ' ' '\n' | grep -E 'CPUTot|RealMemory|CfgTRES' | head
scontrol show node rice-01 2>/dev/null | tr ' ' '\n' | grep -E 'CPUTot|RealMemory|CfgTRES' | head || true
