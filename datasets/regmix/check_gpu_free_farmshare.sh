#!/usr/bin/env bash
# Deprecated: use scripts/farmshare/remote_check_gpu_free.sh (install via sync_check_gpu_free.sh).
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/farmshare/remote_check_gpu_free.sh" "$@"
