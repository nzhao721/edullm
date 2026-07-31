#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<'EOF'
set -e
echo "=== $(date -u +%FT%TZ) ==="
echo "--- scratch top ---"
du -sh /scratch/users/nzhao2 2>/dev/null || true
df -h /scratch/users/nzhao2 2>/dev/null | tail -1 || true

check_raw() {
  local label="$1"
  shift
  echo ""
  echo "=== $label ==="
  for p in "$@"; do
    if [[ -e "$p" ]]; then
      sz=$(du -sh "$p" 2>/dev/null | awk '{print $1}')
      echo "PRESENT  $sz  $p"
      # peek for raw-ish trees
      for sub in data trim raw out documents labels/docs; do
        if [[ -d "$p/$sub" ]]; then
          echo "  sub: $(du -sh "$p/$sub" 2>/dev/null | awk '{print $1}')  $p/$sub"
          find "$p/$sub" -maxdepth 3 \( -name '*.json.gz' -o -name '*.jsonl.gz' -o -name '*.jsonl.zstd' -o -name '*.json.zst' \) 2>/dev/null | head -3 | sed 's/^/    sample: /'
        fi
      done
      # olmo layout data/data
      if [[ -d "$p/data/data" ]]; then
        echo "  sub: $(du -sh "$p/data/data" 2>/dev/null | awk '{print $1}')  $p/data/data"
        find "$p/data/data" -maxdepth 3 \( -name '*.json.gz' -o -name '*.jsonl.gz' \) 2>/dev/null | head -3 | sed 's/^/    sample: /'
      fi
    else
      echo "MISSING  $p"
    fi
  done
}

check_raw "refhq" \
  /scratch/users/nzhao2/refhq-regmix-5p5b-v1 \
  /scratch/users/nzhao2/hq-reference-v1

check_raw "regmix-10b" \
  /scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810 \
  /scratch/users/nzhao2/regmix-10b

# find any regmix / olmo / olmohq / refhq dirs with data/
echo ""
echo "=== scan agent-runs for data/ or trim/ (depth 2) ==="
find /scratch/users/nzhao2/agent-runs -maxdepth 2 -type d \( -name data -o -name trim -o -name raw \) 2>/dev/null | head -40

echo ""
echo "=== named corpus roots under scratch ==="
ls -d /scratch/users/nzhao2/*refhq* /scratch/users/nzhao2/*regmix* /scratch/users/nzhao2/*olmo* /scratch/users/nzhao2/agent-runs/*olmo* /scratch/users/nzhao2/agent-runs/*regmix* /scratch/users/nzhao2/agent-runs/*refhq* /scratch/users/nzhao2/agent-runs/*olmohq* 2>/dev/null | head -60
EOF
