#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
RSH="ssh -S $SOCK -o BatchMode=yes"
SRC=/mnt/c/alpha_ai/edullm

ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
mkdir -p "$STAGING/datasets/olmo" "$STAGING/datasets"
# Ensure shared helpers live where submit scripts expect them.
for f in olmo_shard_utils.py download_s3_shard.py download_s3_shard.sbatch trim_and_tokenize_regmix.py; do
  if [[ -f "$STAGING/datasets/$f" && ! -f "$STAGING/datasets/olmo/../$f" ]]; then
    true
  fi
  # copy from scripts/farmshare dump if present in datasets root
  if [[ -f "$STAGING/scripts/farmshare/$f" && ! -f "$STAGING/datasets/$f" ]]; then
    cp -a "$STAGING/scripts/farmshare/$f" "$STAGING/datasets/$f"
  fi
done
ls "$STAGING/datasets/olmo" | head
ls "$STAGING/datasets/olmohq" | head
test -f "$STAGING/datasets/olmo_shard_utils.py" && echo HAS_UTILS || echo MISSING_UTILS
test -f "$STAGING/datasets/olmo/download_olmo_shard.py" && echo HAS_DL || echo MISSING_DL
test -f "$STAGING/datasets/olmo/tokenize_olmo_shard.py" && echo HAS_TOK || echo MISSING_TOK
REMOTE

rsync -avz -e "$RSH" \
  "$SRC/datasets/olmo/download_olmo_shard.py" \
  "$SRC/datasets/olmo/tokenize_olmo_shard.py" \
  "$SRC/datasets/olmo/build_pool_tokenize_map.py" \
  "$HOST:$STAGING/datasets/olmo/"

rsync -avz -e "$RSH" \
  "$SRC/datasets/olmo_shard_utils.py" \
  "$SRC/datasets/download_s3_shard.py" \
  "$HOST:$STAGING/datasets/"

echo HELPERS_OK
