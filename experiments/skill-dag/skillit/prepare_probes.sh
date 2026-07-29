#!/usr/bin/env bash
# Materialize Skill-It probe slices from an olmohq working pool via mixlaw tooling.
#
# Required:
#   TOKENIZED_DIR   working-pool tokenized/ root (…/tokenized/<domain>/<domain>.npy)
#
# Optional:
#   SKILLIT_PROBE_WORK  local probe work root (default: $WORK/skillit-probes)
#   OUT_DIR         slice output root (default: $SKILLIT_PROBE_WORK/slices)
#   PROBES_JSON     default: sibling probes.json
#   TOKENS_PER_PARAM  default 5
#   WORKERS         parallel build workers (default 4)
#   SEED            default 6198
#   PROBE_ARTIFACTS_S3  published slice-plan prefix
#   S3_EXPORT / SKIP_S3_UPLOAD  disable live artifact sync
#
# Example:
#   TOKENIZED_DIR=$WORK/pool/tokenized SKILLIT_PROBE_WORK=$WORK/skillit-probes \
#     bash prepare_probes.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIXLAW_ROOT="$(cd "${SCRIPT_DIR}/../mixlaw" && pwd)"
PROBES_JSON="${PROBES_JSON:-${SCRIPT_DIR}/probes.json}"
TOKENS_PER_PARAM="${TOKENS_PER_PARAM:-5}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-6198}"
PYTHON="${PYTHON:-python}"
if [[ -z "${SKILLIT_PROBE_WORK:-}" ]]; then
  : "${WORK:?set SKILLIT_PROBE_WORK or WORK}"
  SKILLIT_PROBE_WORK="${WORK%/}/skillit-probes"
fi
OUT_DIR="${OUT_DIR:-${SKILLIT_PROBE_WORK}/slices}"
PROBE_ARTIFACTS_S3="${PROBE_ARTIFACTS_S3:-s3://edullm-datasets/skillit/probes}"

: "${TOKENIZED_DIR:?TOKENIZED_DIR is required}"

export SKILLIT_PROBES_JSON="${PROBES_JSON}"

mkdir -p "${OUT_DIR}"
echo "[prepare_probes] plan from ${PROBES_JSON} tpp=${TOKENS_PER_PARAM}"
"${PYTHON}" "${MIXLAW_ROOT}/build_mixture_data.py" plan \
  --tokenized-dir "${TOKENIZED_DIR}" \
  --out-dir "${OUT_DIR}" \
  --mixtures-json "${PROBES_JSON}" \
  --tokens-per-param "${TOKENS_PER_PARAM}" \
  --seed "${SEED}"

echo "[prepare_probes] build slices → ${OUT_DIR}"
"${PYTHON}" "${MIXLAW_ROOT}/build_mixture_data.py" build \
  --plan-dir "${OUT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --tokenized-dir "${TOKENIZED_DIR}" \
  --workers "${WORKERS}"

S3_EXPORT_VALUE="${S3_EXPORT:-1}"
SKIP_S3_UPLOAD_VALUE="${SKIP_S3_UPLOAD:-}"
S3_EXPORT_VALUE="${S3_EXPORT_VALUE,,}"
SKIP_S3_UPLOAD_VALUE="${SKIP_S3_UPLOAD_VALUE,,}"
if [[ "${S3_EXPORT_VALUE}" =~ ^(0|false|no|off)$ || "${SKIP_S3_UPLOAD_VALUE}" =~ ^(1|true|yes|on)$ ]]; then
  echo "[prepare_probes] S3 export disabled"
elif command -v aws >/dev/null 2>&1; then
  aws s3 sync "${OUT_DIR}" "${PROBE_ARTIFACTS_S3}" --only-show-errors
else
  echo "[prepare_probes] aws CLI missing; probe artifacts remain local" >&2
fi

echo "[prepare_probes] done"
