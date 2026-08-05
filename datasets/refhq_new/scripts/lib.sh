#!/usr/bin/env bash
# Shared helpers for deploying datasets/refhq_new onto a FarmShare run directory.
set -Eeuo pipefail

refhq_new_stage_shared_utils() {
  local staging_root="$1"
  local run_dir="$2"
  mkdir -p "${run_dir}/datasets" "${run_dir}/scripts/farmshare"
  local shared
  for shared in \
    olmo_shard_utils.py \
    trim_and_tokenize_regmix.py \
    edullm_text_companion.py \
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

refhq_new_sync_to_run() {
  local staging_root="$1"
  local run_dir="$2"
  mkdir -p "${run_dir}/datasets"
  cp -a "${staging_root}/datasets/refhq_new" "${run_dir}/datasets/"
  refhq_new_stage_shared_utils "${staging_root}" "${run_dir}"
  sed -i 's/\r$//' "${run_dir}/datasets/refhq_new/scripts/"*.sbatch \
    "${run_dir}/datasets/refhq_new/scripts/"*.sh 2>/dev/null || true
  chmod +x "${run_dir}/datasets/refhq_new/scripts/"*.sbatch \
    "${run_dir}/datasets/refhq_new/scripts/"*.sh 2>/dev/null || true
}

refhq_new_export_pythonpath() {
  local run_dir="$1"
  export PYTHONPATH="${run_dir}/datasets:${PYTHONPATH:-}"
}

refhq_new_load_hf_token() {
  local run_dir="$1"
  if [[ -f "${run_dir}/.hf_token" ]]; then
    HF_TOKEN="$(tr -d '\r\n' < "${run_dir}/.hf_token")"
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
    echo "HF_TOKEN loaded (len=${#HF_TOKEN})"
  else
    echo "WARN: no ${run_dir}/.hf_token (gated HF downloads may fail)" >&2
  fi
}
