#!/usr/bin/env bash
# Sync BLADE / CE-RegMix / REL-EMA checkpoints from sbsandbox S3 to FarmShare scratch.
# Expects AWS_SESSION_ENV (or AWS_* keys) already exported. Prefer DTN for large syncs.
set -Eeuo pipefail

: "${DEST_ROOT:?Set DEST_ROOT to scratch checkpoint root}"
: "${AWS_DEFAULT_REGION:=us-east-1}"

if [[ -n "${AWS_SESSION_ENV:-}" && -f "${AWS_SESSION_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${AWS_SESSION_ENV}"
  unset AWS_PROFILE
fi

: "${AWS_ACCESS_KEY_ID:?}"
: "${AWS_SECRET_ACCESS_KEY:?}"
: "${AWS_SESSION_TOKEN:?}"

export AWS_DEFAULT_REGION AWS_REGION="${AWS_REGION:-$AWS_DEFAULT_REGION}"
command -v aws >/dev/null

mkdir -p \
  "${DEST_ROOT}/blade/checkpoints" \
  "${DEST_ROOT}/ce-regmix/checkpoints" \
  "${DEST_ROOT}/rel-ema"

echo "sync start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "dest=${DEST_ROOT}"

# Parallel syncs (3 prefixes). REL-EMA skips train/ (rank trainer state; not needed for eval).
aws s3 sync \
  "s3://edullm-checkpoints/olmo-370m/edullm-370M-blade-regmix10b/checkpoints/" \
  "${DEST_ROOT}/blade/checkpoints/" \
  --only-show-errors &
pid_blade=$!

aws s3 sync \
  "s3://edullm-checkpoints/olmo-370m/edullm-370M-ce-regmix10b/checkpoints/" \
  "${DEST_ROOT}/ce-regmix/checkpoints/" \
  --only-show-errors &
pid_ce=$!

aws s3 sync \
  "s3://edullm-checkpoints/token-selection/rel-ema-10b-scratch-v1/rel_ema/" \
  "${DEST_ROOT}/rel-ema/" \
  --exclude "train/*" \
  --exclude "*/train/*" \
  --exclude "run_fingerprint.json" \
  --only-show-errors &
pid_rel=$!

ec=0
for pid in "${pid_blade}" "${pid_ce}" "${pid_rel}"; do
  if ! wait "${pid}"; then
    ec=1
  fi
done

echo "sync end $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=${ec}"
du -sh \
  "${DEST_ROOT}/blade" \
  "${DEST_ROOT}/ce-regmix" \
  "${DEST_ROOT}/rel-ema" \
  "${DEST_ROOT}" 2>/dev/null || true

# Summarize step dirs
for name in blade/checkpoints ce-regmix/checkpoints rel-ema; do
  echo "-- ${name} --"
  ls -1 "${DEST_ROOT}/${name}" 2>/dev/null | head -40 || true
done

exit "${ec}"
