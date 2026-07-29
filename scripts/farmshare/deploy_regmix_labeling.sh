#!/usr/bin/env bash
set -Eeuo pipefail

SOCK="${SOCK:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
RUN="${RUN:-/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810}"
REPO="${REPO:-/mnt/c/alpha_ai/edullm}"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "mkdir -p '${RUN}/scripts' '${RUN}/logs' '${RUN}/labels'"

SCP_OPTS=(-o "ControlPath=${SOCK}")

for f in text_difficulty_metrics.py label_olmo_shard.py finalize_olmo_labels.py materialize_curriculum.py; do
  scp "${SCP_OPTS[@]}" "${REPO}/datasets/olmo/${f}" "${HOST}:${RUN}/scripts/${f}"
done

for f in build_regmix_label_manifest.py label_regmix_shard.sbatch submit_regmix_labeling.sh; do
  scp "${SCP_OPTS[@]}" "${REPO}/datasets/regmix/${f}" "${HOST}:${RUN}/scripts/${f}"
done

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "chmod +x '${RUN}/scripts/'*.py '${RUN}/scripts/'*.sh '${RUN}/scripts/'*.sbatch 2>/dev/null || true"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "cd '${RUN}' && bash scripts/submit_regmix_labeling.sh"
