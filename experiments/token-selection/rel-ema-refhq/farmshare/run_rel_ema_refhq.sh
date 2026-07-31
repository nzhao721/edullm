#!/usr/bin/env bash
# Thin FarmShare wrapper — same hardware-agnostic launcher, default scratch paths.
# Ephemeral: RUN_DIR starts empty; stage edullm-data; durable export via spine.
# Does not hardcode GPU indices or node names. Pass NUM_GPUS / CUDA_VISIBLE_DEVICES.
# REF_PT optional: if unset, --launch auto-materializes reference.s3_uri from S3.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_DIR="${RUN_DIR:-${SCRATCH:-/scratch}/rel-ema-refhq-10b}"
export RANK_MICROBATCH_SIZE="${RANK_MICROBATCH_SIZE:-16384}"
exec bash "$SCRIPT_DIR/../launch_train.sh" "$@"
