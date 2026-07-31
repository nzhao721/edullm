#!/usr/bin/env bash
# Resume SmolLM2 DDP from a durable S3 checkpoint (ephemeral scratch safe).
#
# Required:
#   RESUME_FROM_S3=s3://edullm-checkpoints/smollm2/<prior-run>/checkpoints/stepNNNNNNN/
# Optional:
#   RUN_NAME, CHECKPOINT_S3_URI, DATASET_ID, NUM_NODES, ...
#
# Do not point RESUME_FROM at a wiped scratch path.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUNET="${SUNET:-${USER:-nzhao2}}"

if [[ -z "${RESUME_FROM_S3:-}" ]]; then
  echo "RESUME_FROM_S3 is required (durable checkpoint URI). Local scratch resume is unsupported." >&2
  exit 2
fi

export SUNET
export RESUME_FROM_S3
unset RESUME_FROM
export TRAIN_PY="${TRAIN_PY:-${SCRIPT_DIR}/train_smollm2_135m_ddp.py}"
bash "${SCRIPT_DIR}/submit_smollm2_135m_500m_40ep.sh"
