#!/usr/bin/env bash
# Compatibility wrapper — prefer launch_train.sh (matches control / learnability-doc).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/launch_train.sh" "$@"
