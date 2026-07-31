#!/usr/bin/env bash
# RETIRED: slicing a persistent FarmShare FineWeb memmap for local DDP training.
# Publish tokenized corpora to s3://edullm-data and train via submit_smollm2_135m_500m_40ep.sh.
set -Eeuo pipefail
echo "submit_slice_fineweb_edu_750m.sh is retired (persistent scratch slices)." >&2
echo "Publish under edullm-data, then: bash scripts/farmshare/submit_smollm2_135m_500m_40ep.sh" >&2
exit 2
