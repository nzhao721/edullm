#!/usr/bin/env bash
# Shared helpers for deploying datasets/refhq onto a FarmShare run directory.
set -Eeuo pipefail

dolma_hq_stage_shared_utils() {
  local staging_root="$1"
  local run_dir="$2"
  mkdir -p "${run_dir}/datasets" "${run_dir}/scripts/farmshare"
  local shared
  for shared in \
    olmo_shard_utils.py \
    trim_and_tokenize_regmix.py \
    download_s3_shard.py; do
    if [[ -f "${staging_root}/datasets/${shared}" ]]; then
      cp -a "${staging_root}/datasets/${shared}" "${run_dir}/datasets/"
    fi
  done
  for infra in \
    prepare_aws_session_light.sh \
    write_aws_session_env.py; do
    if [[ -f "${staging_root}/scripts/farmshare/${infra}" ]]; then
      cp -a "${staging_root}/scripts/farmshare/${infra}" "${run_dir}/scripts/farmshare/"
    fi
  done
}

dolma_hq_sync_to_run() {
  local staging_root="$1"
  local run_dir="$2"
  mkdir -p "${run_dir}/datasets"
  cp -a "${staging_root}/datasets/refhq" "${run_dir}/datasets/"
  dolma_hq_stage_shared_utils "${staging_root}" "${run_dir}"
  sed -i 's/\r$//' "${run_dir}/datasets/refhq/scripts/"*.sbatch "${run_dir}/datasets/refhq/scripts/"*.sh 2>/dev/null || true
  chmod +x "${run_dir}/datasets/refhq/scripts/"*.sbatch "${run_dir}/datasets/refhq/scripts/"*.sh 2>/dev/null || true
}

dolma_hq_export_pythonpath() {
  local run_dir="$1"
  export PYTHONPATH="${run_dir}/datasets:${PYTHONPATH:-}"
}
