#!/usr/bin/env bash
# Thin FarmShare wrapper — same hardware-agnostic launcher, default scratch paths.
# Ephemeral: RUN_DIR starts empty; stage edullm-data; durable export via spine.
# Does not hardcode GPU indices or node names. Pass NUM_GPUS / CUDA_VISIBLE_DEVICES.
# RANK_MICROBATCH_SIZE defaults to 16384 for 48 GiB L40S; YAML canonical is 65536.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_DIR="${RUN_DIR:-${SCRATCH:-/scratch}/rel-ema-exp-10b}"
export RANK_MICROBATCH_SIZE="${RANK_MICROBATCH_SIZE:-16384}"
exec bash "$SCRIPT_DIR/../launch_train.sh" "$@"
