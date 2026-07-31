#!/usr/bin/env bash
# Fresh SmolLM2 DDP submit (no local FineWeb slice). Prefer sync_submit from the laptop.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUNET="${SUNET:-${USER:-nzhao2}}"
export SUNET
export NODELIST="${NODELIST:-}"
bash "${SCRIPT_DIR}/submit_smollm2_135m_500m_40ep.sh"
sleep 2
squeue -u "${SUNET}"
