#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<'EOF'
set -e
for p in \
  /scratch/users/nzhao2/agent-runs/olmo-mix-30b-20260722 \
  /scratch/users/nzhao2/agent-runs/olmo-mix-30b-20260722/data \
  /scratch/users/nzhao2/agent-runs/olmo-mix-30b-20260722/trim \
  /scratch/users/nzhao2/agent-runs/olmo-mix-upsample-20260723-103547 \
  /scratch/users/nzhao2/agent-runs/olmo-mix-upsample-20260723-103547/data \
  /scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841 \
  /scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841/data \
  /scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z/publish-stage \
  /scratch/users/nzhao2/agent-runs/olmo30b-edullm-publish-20260731T001430Z/publish-stage
do
  if [[ -e "$p" ]]; then
    echo "PRESENT $(du -sh "$p" 2>/dev/null | awk '{print $1}') $p"
  else
    echo "MISSING $p"
  fi
done
echo "--- sample json under 30b data ---"
find /scratch/users/nzhao2/agent-runs/olmo-mix-30b-20260722/data -maxdepth 4 \( -name '*.json.gz' -o -name '*.jsonl.gz' \) 2>/dev/null | head -5
echo "--- sample json under upsample data ---"
find /scratch/users/nzhao2/agent-runs/olmo-mix-upsample-20260723-103547/data -maxdepth 4 \( -name '*.json.gz' -o -name '*.jsonl.gz' \) 2>/dev/null | head -5
EOF
