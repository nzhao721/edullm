#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
echo "=== summary ==="
for j in 1666301 1666360 1666357 1666358 1666359; do
  # Prefer array task rows (contain _), else parent job row.
  sacct -j "$j" --format=JobID,JobName%16,State,ExitCode,Elapsed -n -P 2>/dev/null \
    | awk -F'|' -v j="$j" '
      $1 ~ /^[0-9]+_[0-9]+$/ {c[$3]++; n++}
      $1 == j {parent=$3; pel=$5; pe=$4}
      END {
        if (n>0) {
          printf "job %s array tasks: ", j
          for (s in c) printf "%s=%d ", s, c[s]
          print ""
        } else {
          printf "job %s parent: state=%s exit=%s elapsed=%s\n", j, parent, pe, pel
        }
      }'
done
echo "=== pool ==="
tail -n 15 /scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236/logs/pool-1666357.out 2>/dev/null || true
echo "=== slice ==="
tail -n 20 /scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236/logs/slice-1666358.out 2>/dev/null || true
tail -n 10 /scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236/logs/slice-1666358.err 2>/dev/null || true
echo "=== topup up ==="
tail -n 15 /scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841/logs/upload-1666360.out 2>/dev/null || true
REMOTE
