#!/usr/bin/env bash
# Deprecated: difficulty labels now target RegMix 10B, not the ~30B OLMo sample.
set -Eeuo pipefail

echo "submit_olmo_labeling.sh is deprecated." >&2
echo "Use datasets/regmix/submit_regmix_labeling.sh for RegMix 10B labeling." >&2
exit 1
