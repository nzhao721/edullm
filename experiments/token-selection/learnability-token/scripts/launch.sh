#!/usr/bin/env bash
# Compat wrapper — prefer learnability-token/launch.sh (arm root).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../launch.sh" "$@"
