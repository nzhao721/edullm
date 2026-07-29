#!/usr/bin/env bash
# Compatibility wrapper — canonical launcher lives at ../launch_train.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../launch_train.sh" "$@"
