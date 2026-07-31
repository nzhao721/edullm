#!/usr/bin/env bash
# RETIRED: local FineWeb memmap / scratch-slice training.
# Use scripts/farmshare/submit_smollm2_135m_500m_40ep.sh (stages s3://edullm-data, durable S3/W&B).
set -Eeuo pipefail
echo "submit_smollm2_local_750m.sh is retired (assumed persistent scratch slices)." >&2
echo "Use: bash scripts/farmshare/submit_smollm2_135m_500m_40ep.sh" >&2
exit 2
