#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=fw1b-combined
#SBATCH --output=logs/fineweb_edu_1b_combined_%j.out
#SBATCH --error=logs/fineweb_edu_1b_combined_%j.err

set -Eeuo pipefail
: "${RUN_DIR:?}" "${STAGE_DIR:?}" "${AWS_SESSION_ENV:?}" "${EDULLM_ROOT:?}"

VENV="${VENV:-${RUN_DIR}/venv}"

mkdir -p "${RUN_DIR}/logs"
cd "${RUN_DIR}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV}"
export PATH="${PATH}:${HOME}/.local/bin:${HOME}/tools/aws/bin"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
unset AWS_PROFILE || true

WHEEL_URI="${WHEEL_URI:-s3://edullm-landing/_dist/edullm_data-0.6.3-py3-none-any.whl}"
NEWEST="$(aws s3 ls s3://edullm-landing/_dist/ 2>/dev/null | awk '/edullm_data-.*\.whl$/ {print $NF}' | sort -V | tail -1 || true)"
if [[ -n "${NEWEST}" ]]; then
  WHEEL_URI="s3://edullm-landing/_dist/${NEWEST}"
fi

pip install -q -U pip wheel boto3
WHEEL_BASENAME="$(basename "${WHEEL_URI}")"
aws s3 cp "${WHEEL_URI}" "${RUN_DIR}/${WHEEL_BASENAME}"
pip install -q --force-reinstall "${RUN_DIR}/${WHEEL_BASENAME}"
python -c 'import edullm_data; v=getattr(edullm_data,"__version__","?"); print(v, edullm_data.__file__); assert not str(v).startswith("0.2"), v'

export RUN_DIR STAGE_DIR AWS_SESSION_ENV
export TOKENIZED_ROOT="${TOKENIZED_ROOT:-/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-tokenized}"
export RAW_ROOT="${RAW_ROOT:-/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-raw}"
export HF_FINEWEB_EDU_REVISION="${HF_FINEWEB_EDU_REVISION:-87f09149ef4734204d70ed1d046ddc9ca3f2b8f9}"

python "${EDULLM_ROOT}/datasets/fineweb/publish_fineweb_edu_1b_combined.py"
