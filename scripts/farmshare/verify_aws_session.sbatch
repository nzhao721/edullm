#!/usr/bin/env bash
# Minimal normal-partition verification for FarmShare's login-to-compute AWS handoff.
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --exclude=wheat-01

set -Eeuo pipefail

: "${EDULLM_ROOT:?}" "${RUN_DIR:?}" "${PIPELINE_BUCKET:?}" "${PIPELINE_PREFIX:?}"
: "${AWS_SESSION_ENV:=${RUN_DIR}/aws-session.env}"
# shellcheck source=bootstrap.sh
source "${EDULLM_ROOT}/scripts/farmshare/bootstrap.sh"

probe_key="${PIPELINE_PREFIX}/_farmshare_aws_probe/${SLURM_JOB_ID}.txt"
probe_uri="s3://${PIPELINE_BUCKET}/${probe_key}"
cleanup() {
  aws s3 rm "${probe_uri}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

aws sts get-caller-identity --output json >/dev/null
printf 'farmshare-aws-session-probe\n' | aws s3 cp - "${probe_uri}" >/dev/null
[[ "$(aws s3 cp "${probe_uri}" -)" == "farmshare-aws-session-probe" ]]
aws s3 rm "${probe_uri}" >/dev/null
trap - EXIT
echo "FarmShare AWS compute verification passed"
