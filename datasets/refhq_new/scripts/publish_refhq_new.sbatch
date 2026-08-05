#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=refhq-new-pub
#SBATCH --output=logs/refhq_new_publish_%j.out
#SBATCH --error=logs/refhq_new_publish_%j.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${STAGE_DIR:?}"

VENV="${VENV:-${RUN_DIR}/venv}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [[ -n "${AWS_SESSION_ENV:-}" && -f "${AWS_SESSION_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${AWS_SESSION_ENV}"
elif [[ -f "${RUN_DIR}/aws-session.env" ]]; then
  # shellcheck disable=SC1091
  source "${RUN_DIR}/aws-session.env"
  export AWS_SESSION_ENV="${RUN_DIR}/aws-session.env"
else
  echo "ERROR: AWS_SESSION_ENV missing; push via scripts/farmshare/push_aws_session_to_farmshare.sh ${RUN_DIR}" >&2
  exit 1
fi

export PATH="${PATH}:${HOME}/.local/bin:${HOME}/tools/aws/bin"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
unset AWS_PROFILE || true

: "${REFHQ_NEW_SCRIPTS:=${RUN_DIR}/datasets/refhq_new/scripts}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${RUN_DIR}}"
TOKENIZED_ROOT="${TOKENIZED_ROOT:-}"
MANIFEST="${MANIFEST:-${SCRATCH_ROOT}/manifests/tokenized_manifest.json}"
TEXT_RUN_DIR="${TEXT_RUN_DIR:-${SCRATCH_ROOT}}"

# shellcheck disable=SC1091
source "${REFHQ_NEW_SCRIPTS}/lib.sh"
refhq_new_export_pythonpath "${RUN_DIR}"

# Always newest edullm-data: prefer landing _dist wheel, else git@main (never pin old tags).
WHEEL_URI="${WHEEL_URI:-}"
NEWEST="$(aws s3 ls s3://edullm-landing/_dist/ 2>/dev/null | awk '/edullm_data-.*\.whl$/ {print $NF}' | sort -V | tail -1 || true)"
if [[ -n "${NEWEST}" ]]; then
  WHEEL_URI="s3://edullm-landing/_dist/${NEWEST}"
fi
pip install -q -U pip wheel boto3
if [[ -n "${WHEEL_URI}" ]]; then
  WHEEL_BASENAME="$(basename "${WHEEL_URI}")"
  aws s3 cp "${WHEEL_URI}" "${RUN_DIR}/${WHEEL_BASENAME}"
  pip install -q --force-reinstall "${RUN_DIR}/${WHEEL_BASENAME}"
else
  pip install -q --force-reinstall \
    "edullm-data @ git+https://github.com/edu-llm/edullm-data@main"
fi
python -c 'import edullm_data; print("edullm_data", getattr(edullm_data,"__version__","?"), edullm_data.__file__)'

PUB_ARGS=(
  --scratch-root "${SCRATCH_ROOT}"
  --stage-dir "${STAGE_DIR}"
  --text-run-dir "${TEXT_RUN_DIR}"
  --dataset-id "${DATASET_ID:-pretrain/refhq-instruct}"
  --hash-workers 8
  --copy-workers 8
  --force
)
if [[ -n "${TOKENIZED_ROOT}" ]]; then
  PUB_ARGS+=(--tokenized-root "${TOKENIZED_ROOT}")
fi
if [[ -f "${MANIFEST}" ]]; then
  PUB_ARGS+=(--manifest "${MANIFEST}")
fi
# finalize already staged tokens/ into STAGE_DIR; re-stage text only if needed.
if [[ -d "${STAGE_DIR}/tokens" ]]; then
  PUB_ARGS+=(--skip-stage)
fi
# text/ or vendor/ already staged from prior attempt
if [[ -d "${STAGE_DIR}/text" || -d "${STAGE_DIR}/vendor" ]]; then
  PUB_ARGS+=(--skip-text-stage)
fi

python "${REFHQ_NEW_SCRIPTS}/publish_refhq_new.py" "${PUB_ARGS[@]}"
