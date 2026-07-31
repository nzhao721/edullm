#!/usr/bin/env bash
set -Eeuo pipefail

SOCKET="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
SUNET="${SUNET:-nzhao2}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/regmix-10b-20260725-124810}"
VENV="${VENV:-/scratch/users/${SUNET}/agent-runs/olmo-ladder-370m-20260722-185217/venv}"

ssh -S "${SOCKET}" -o BatchMode=yes "${SUNET}@login.farmshare.stanford.edu" \
  "set -Eeuo pipefail; source '${VENV}/bin/activate'; export PYTHONPATH='${RUN_DIR}/scripts'; python '${RUN_DIR}/scripts/finalize_regmix_lm_labels.py' --labels-root '${RUN_DIR}/lm_labels/labels' --work-manifest '${RUN_DIR}/lm_labels/lm_work_manifest.jsonl' && cat '${RUN_DIR}/lm_labels/labels/READY'"
