#!/usr/bin/env bash
# Submit per-domain stream-faithful n_tokens rebuild jobs on FarmShare.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
REGMIX_ROOT="${REGMIX_ROOT:-/scratch/users/${SUNET}/agent-runs/regmix-10b-20260725-124810}"
STAGING_ROOT="${STAGING_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/regmix-stream-ntokens-$(date -u +%Y%m%dT%H%M%SZ)}"
DOMAINS="${DOMAINS:-algebraic-stack arxiv dclm open-web-math pes2o starcoder wiki}"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/scripts" "${RUN_DIR}/stream_lengths" "${RUN_DIR}/lm_labels_stream_aligned"
cd "${RUN_DIR}"

if [[ -d "${REGMIX_ROOT}/venv" && ! -e "${RUN_DIR}/venv" ]]; then
  ln -s "${REGMIX_ROOT}/venv" "${RUN_DIR}/venv"
fi
# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"

cp -a "${STAGING_ROOT}/datasets/regmix/rebuild_stream_n_tokens.py" "${RUN_DIR}/scripts/"
cp -a "${STAGING_ROOT}/datasets/regmix/rebuild_stream_n_tokens.sbatch" "${RUN_DIR}/scripts/"
sed -i 's/\r$//' "${RUN_DIR}/scripts/"*.py "${RUN_DIR}/scripts/"*.sbatch || true

for domain in ${DOMAINS}; do
  JOB=$(sbatch --parsable --exclude=wheat-01 \
    --chdir="${RUN_DIR}" \
    --job-name="regmix-ntok-${domain}" \
    --export=ALL,RUN_DIR="${RUN_DIR}",REGMIX_ROOT="${REGMIX_ROOT}",SCRIPTS="${RUN_DIR}/scripts",DOMAIN="${domain}",BATCH_SIZE="${BATCH_SIZE:-256}" \
    "${RUN_DIR}/scripts/rebuild_stream_n_tokens.sbatch")
  echo "domain=${domain} job=${JOB}"
done
echo "run_dir=${RUN_DIR}"
