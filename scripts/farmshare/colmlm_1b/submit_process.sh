#!/usr/bin/env bash
# Keep submit_process.sh as a convenience wrapper (both stages together).
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
SPAN_WORKERS="${SPAN_WORKERS:-256}"
SPAN_THROTTLE="${SPAN_THROTTLE:-128}"
RUN_DIR="${RUN_DIR}" SCRIPT_DIR="${SCRIPT_DIR}" bash "${SCRIPT_DIR}/submit_docs.sh"
RUN_DIR="${RUN_DIR}" SCRIPT_DIR="${SCRIPT_DIR}" \
  SPAN_WORKERS="${SPAN_WORKERS}" SPAN_THROTTLE="${SPAN_THROTTLE}" \
  bash "${SCRIPT_DIR}/submit_spans_and_mark.sh"
