#!/usr/bin/env bash
set -euo pipefail

SOCKET="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
SUNET="${SUNET:-nzhao2}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/regmix-10b-20260725-124810}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ssh_cmd() {
  ssh -S "${SOCKET}" -o BatchMode=yes "${SUNET}@login.farmshare.stanford.edu" "$@"
}

ssh_cmd "mkdir -p '${RUN_DIR}/scripts' '${RUN_DIR}/logs'"

for name in \
  build_regmix_lm_retry_indices.py \
  label_regmix_doc_lm.py \
  label_regmix_doc_lm_retry.sbatch \
  control_regmix_doc_lm_retry.sbatch \
  control_regmix_doc_lm.sbatch \
  finalize_regmix_lm_labels.py \
  submit_regmix_doc_lm_retry.sh \
  check_lm_progress.sh; do
  ssh_cmd "cat > '${RUN_DIR}/scripts/${name}'" < "${SCRIPT_DIR}/${name}"
done

ssh_cmd "chmod +x '${RUN_DIR}/scripts/'*.py '${RUN_DIR}/scripts/'*.sh '${RUN_DIR}/scripts/'*.sbatch 2>/dev/null || true"

if [[ "${SUBMIT:-0}" == "1" ]]; then
  echo "submit pipeline retry (overlapping arrays)"
  ssh_cmd "bash '${RUN_DIR}/scripts/submit_regmix_doc_lm_retry.sh'"
else
  echo "deploy only (set SUBMIT=1 to submit controller)"
fi
