#!/usr/bin/env bash
# Copy only the code required by the text-corpus/v1 republish jobs to FarmShare.
set -Eeuo pipefail

DEST="${1:?usage: $0 /scratch/users/SUNET/run/edullm}"
SOCK="${FARMSHARE_SOCK:-/tmp/farmshare-nzhao2.sock}"
HOST="${FARMSHARE_HOST:-nzhao2@login.farmshare.stanford.edu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

tar -C "${ROOT}" -czf - \
  datasets/edullm_text_companion.py \
  datasets/olmo_shard_utils.py \
  datasets/trim_and_tokenize_regmix.py \
  datasets/regmix/publish_regmix_edullm_data.py \
  datasets/regmix/publish_regmix_edullm_data.sbatch \
  datasets/refhq/scripts/publish_refhq_edullm_data.py \
  datasets/refhq/scripts/publish_refhq_edullm_data.sbatch \
  datasets/olmo/publish_olmo_original_30b_edullm_data.sbatch \
  datasets/olmohq/publish_olmohq_edullm_data.py \
  datasets/olmohq/publish_olmohq_edullm_data.sbatch \
  datasets/olmohq/publish_olmohq_skip_stage.sbatch |
  ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
    "mkdir -p $(printf '%q' "${DEST}") && tar -xzf - -C $(printf '%q' "${DEST}")"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "test -f $(printf '%q' "${DEST}/datasets/edullm_text_companion.py")"
echo "synced text-corpus publish code to ${DEST}"
