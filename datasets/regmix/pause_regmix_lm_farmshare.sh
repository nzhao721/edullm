#!/usr/bin/env bash
set -euo pipefail

SOCKET="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
SUNET="${SUNET:-nzhao2}"

ssh -S "${SOCKET}" -o BatchMode=yes "${SUNET}@login.farmshare.stanford.edu" bash <<REMOTE
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
for name in regmix-lm-pipe regmix-lm-retry regmix-lm-chunk regmix-lm; do
  n=\$(squeue -u ${SUNET} -h -n "\${name}" 2>/dev/null | wc -l | tr -d ' ')
  if [[ "\${n}" -gt 0 ]]; then
    echo "scancel -n \${name} (\${n} jobs)"
    scancel -n "\${name}" -u ${SUNET} 2>/dev/null || true
  fi
done
sleep 3
echo "remaining regmix jobs:"
squeue -u ${SUNET} 2>/dev/null | grep regmix || echo none
ids=\$(squeue -u ${SUNET} -h -o "%i %j" 2>/dev/null | awk '/regmix/{print \$1}')
if [[ -n "\${ids}" ]]; then
  echo "force scancel remaining: \${ids}"
  scancel \${ids} 2>/dev/null || true
  sleep 2
  squeue -u ${SUNET} 2>/dev/null | grep regmix || echo none
fi
REMOTE
