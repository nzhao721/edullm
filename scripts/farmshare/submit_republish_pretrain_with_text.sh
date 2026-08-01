#!/usr/bin/env bash
# Submit FarmShare republish jobs: tokens + text-corpus/v1 companions → edullm-landing.
#
# Prereqs (on laptop):
#   1. Open control socket: ssh -M -S /tmp/farmshare-${SUNET}.sock -o ControlPersist=yes ${SUNET}@login.farmshare.stanford.edu
#   2. Mint + push AWS session: scripts/farmshare/push_aws_session_to_farmshare.sh "${RUN_DIR}"
#   3. Start refresh loop: scripts/farmshare/loop_push_aws_session_to_farmshare.sh "${RUN_DIR}"
#
# Each job writes a new dataset version (v2+) with groups tokens/ + text/.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
SOCK="${FARMSSH_SOCK:-/tmp/farmshare-${SUNET}.sock}"
HOST="${SUNET}@login.farmshare.stanford.edu"
EDULLM_ROOT="${EDULLM_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging/edullm}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<EOF
set -Eeuo pipefail
export SUNET=${SUNET}
export EDULLM_ROOT=${EDULLM_ROOT}
export STAMP=${STAMP}

submit_one() {
  local name="\$1" run_dir="\$2" sbatch_rel="\$3"
  shift 3
  mkdir -p "\${run_dir}/logs"
  cd "\${run_dir}"
  # shellcheck disable=SC2086
  sbatch --exclude=wheat-01 "\${EDULLM_ROOT}/datasets/\${sbatch_rel}" "\$@"
  echo "submitted \${name} run_dir=\${run_dir}"
}

# RegMix 10B
submit_one regmix-10b \\
  "/scratch/users/\${SUNET}/agent-runs/regmix-10b-text-\${STAMP}" \\
  regmix/publish_regmix_edullm_data.sbatch \\
  --export=ALL,RUN_DIR="/scratch/users/\${SUNET}/agent-runs/regmix-10b-text-\${STAMP}",REGMIX_ROOT="/scratch/users/\${SUNET}/agent-runs/regmix-10b-20260725-124810",STAGE_DIR="/scratch/users/\${SUNET}/agent-runs/regmix-10b-text-\${STAMP}/publish-stage",AWS_SESSION_ENV="/scratch/users/\${SUNET}/agent-runs/regmix-10b-text-\${STAMP}/aws-session.env"

# RefHQ 5.5B
submit_one refhq-regmix-5p5b \\
  "/scratch/users/\${SUNET}/agent-runs/refhq-text-\${STAMP}" \\
  refhq/scripts/publish_refhq_edullm_data.sbatch \\
  --export=ALL,RUN_DIR="/scratch/users/\${SUNET}/agent-runs/refhq-text-\${STAMP}",REFHQ_ROOT="/scratch/users/\${SUNET}/refhq-regmix-5p5b-v1",STAGE_DIR="/scratch/users/\${SUNET}/agent-runs/refhq-text-\${STAMP}/publish-stage",AWS_SESSION_ENV="/scratch/users/\${SUNET}/agent-runs/refhq-text-\${STAMP}/aws-session.env"

# OLMo original 30B
submit_one olmo-original-30b \\
  "/scratch/users/\${SUNET}/agent-runs/olmo30b-text-\${STAMP}" \\
  olmo/publish_olmo_original_30b_edullm_data.sbatch \\
  --export=ALL,RUN_DIR="/scratch/users/\${SUNET}/agent-runs/olmo30b-text-\${STAMP}",STAGE_DIR="/scratch/users/\${SUNET}/agent-runs/olmo30b-text-\${STAMP}/publish-stage",AWS_SESSION_ENV="/scratch/users/\${SUNET}/agent-runs/olmo30b-text-\${STAMP}/aws-session.env",TEXT_RUN_DIR="/scratch/users/\${SUNET}/agent-runs/olmo-mix-30b-20260722"

# OLMo ~127B (reuse prior token stage if present; only stages text + publish)
PRIOR_STAGE="/scratch/users/\${SUNET}/agent-runs/olmo127b-edullm-publish-20260730T233445Z/publish-stage"
submit_one olmo-127b \\
  "/scratch/users/\${SUNET}/agent-runs/olmo127b-text-\${STAMP}" \\
  olmohq/publish_olmohq_skip_stage.sbatch \\
  --export=ALL,RUN_DIR="/scratch/users/\${SUNET}/agent-runs/olmo127b-text-\${STAMP}",STAGE_DIR="\${PRIOR_STAGE}",AWS_SESSION_ENV="/scratch/users/\${SUNET}/agent-runs/olmo127b-text-\${STAMP}/aws-session.env",TEXT_RUN_DIR="/scratch/users/\${SUNET}/agent-runs/olmo-mix-upsample-20260723-103547"
EOF
