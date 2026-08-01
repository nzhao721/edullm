#!/usr/bin/env bash
# Stage RegMix curriculum token-order rankings and publish to edullm-landing -> edullm-data.
#
# Prerequisite: parent_pool_flat_chunks_v1 build output under INDEX_DIR.
# Target dataset: curriculum/regmix-370m (four token-order/v1 groups).
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
REGMIX_ROOT="${REGMIX_ROOT:-/scratch/users/${SUNET}/agent-runs/regmix-10b-20260725-124810}"
STAGING_ROOT="${STAGING_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/regmix-curriculum-edullm-publish-$(date -u +%Y%m%dT%H%M%SZ)}"
STAGE_DIR="${STAGE_DIR:-${RUN_DIR}/publish-stage}"
INDEX_DIR="${INDEX_DIR:-${REGMIX_ROOT}/curriculum_index}"
DRY_RUN="${DRY_RUN:-0}"
PARENT_VERSION="${PARENT_VERSION:-}"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/scripts/regmix" "${RUN_DIR}/scripts/curriculum" "${RUN_DIR}/scripts/farmshare"
cd "${RUN_DIR}"

if [[ -d "${REGMIX_ROOT}/venv" && ! -e "${RUN_DIR}/venv" ]]; then
  ln -s "${REGMIX_ROOT}/venv" "${RUN_DIR}/venv"
fi
if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
  # shellcheck disable=SC1091
  source "${RUN_DIR}/venv/bin/activate"
  pip install -U pip wheel boto3 numpy
else
  # shellcheck disable=SC1091
  source "${RUN_DIR}/venv/bin/activate"
fi

cp -a "${STAGING_ROOT}/datasets/regmix/publish_regmix_curriculum_edullm_data.py" "${RUN_DIR}/scripts/regmix/"
cp -a "${STAGING_ROOT}/datasets/regmix/publish_regmix_curriculum_edullm_data.sbatch" "${RUN_DIR}/scripts/regmix/"
cp -a "${STAGING_ROOT}/experiments/curriculum/curriculum_pacing.py" "${RUN_DIR}/scripts/curriculum/"
sed -i 's/\r$//' \
  "${RUN_DIR}/scripts/regmix/"*.py \
  "${RUN_DIR}/scripts/regmix/"*.sbatch \
  "${RUN_DIR}/scripts/curriculum/"*.py \
  2>/dev/null || true

missing=0
for metric in compression_ratio flesch mtld learnability; do
  if [[ ! -f "${INDEX_DIR}/ranked_chunks_${metric}.npy" ]]; then
    echo "ERROR: missing ${INDEX_DIR}/ranked_chunks_${metric}.npy" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "Build the index first, e.g.:" >&2
  echo "  python experiments/curriculum/scripts/build_curriculum_index.py \\" >&2
  echo "    --labels-root ${REGMIX_ROOT}/labels --lm-labels-root ${REGMIX_ROOT}/lm_labels/labels \\" >&2
    echo "    --parent-layout /path/to/pinned-parent-layout.json \\" >&2
    echo "    --parent-version vN --parent-manifest-sha256 SHA256 \\" >&2
    echo "    --out-dir ${INDEX_DIR}" >&2
  exit 1
fi

export EDULLM_ROOT="${RUN_DIR}"
export RUN_DIR INDEX_DIR STAGE_DIR DRY_RUN PARENT_VERSION
if [[ ! -s "${AWS_SESSION_ENV:-}" ]]; then
  echo "ERROR: AWS_SESSION_ENV must be a pre-minted, job-scoped session file." >&2
  echo "Mint on the engineer device and push it with" >&2
  echo "scripts/farmshare/push_aws_session_to_farmshare.sh ${RUN_DIR}" >&2
  exit 1
fi

JOB=$(sbatch --parsable --exclude=wheat-01 \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",INDEX_DIR="${INDEX_DIR}",STAGE_DIR="${STAGE_DIR}",EDULLM_ROOT="${EDULLM_ROOT}",SCRIPTS="${RUN_DIR}/scripts/regmix",AWS_SESSION_ENV="${AWS_SESSION_ENV}",DRY_RUN="${DRY_RUN}",PARENT_VERSION="${PARENT_VERSION}" \
  "${RUN_DIR}/scripts/regmix/publish_regmix_curriculum_edullm_data.sbatch")
echo "publish_job=${JOB}"
echo "run_dir=${RUN_DIR}"
echo "index_dir=${INDEX_DIR}"
echo "stage_dir=${STAGE_DIR}"
echo "target_dataset=curriculum/regmix-370m"
echo "dry_run=${DRY_RUN}"
if [[ -n "${PARENT_VERSION}" ]]; then
  echo "parent_version=${PARENT_VERSION}"
fi
