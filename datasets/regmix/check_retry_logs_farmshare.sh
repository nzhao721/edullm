#!/usr/bin/env bash
SOCKET=/tmp/farmshare-nzhao2.sock
ssh -S "$SOCKET" -o BatchMode=yes nzhao2@login.farmshare.stanford.edu bash <<'REMOTE'
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
RUN=/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810
echo recent_success_logs:
ls -t "$RUN/logs/lm-retry-"*.out 2>/dev/null | head -5 | while read -r f; do
  if grep -q labeled_lm_chunk "$f" 2>/dev/null; then
    echo OK "$f" "$(grep labeled_lm_chunk "$f" | tail -1 | cut -c1-120)"
  else
    echo NO "$f" "$(tail -1 "$f" 2>/dev/null | cut -c1-80)"
  fi
done
echo failed_sample:
ls -t "$RUN/logs/lm-retry-"*.err 2>/dev/null | head -3 | while read -r f; do
  echo "--- $f ---"
  tail -3 "$f"
done
REMOTE
