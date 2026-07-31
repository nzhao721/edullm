#!/usr/bin/env bash
# Cancel 1670742 and resubmit skip-stage at 80G.
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
REPO=/mnt/c/alpha_ai/edullm
RUN=/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging

# Confirm mem line
grep -n 'mem=' "$REPO/datasets/olmohq/publish_olmohq_skip_stage.sbatch"

sed -i 's/\r$//' \
  "$REPO/datasets/olmohq/publish_olmohq_edullm_data.py" \
  "$REPO/datasets/olmohq/publish_olmohq_skip_stage.sbatch"

# Refresh creds + restart 10m laptop pushe
pkill -f loop_push_aws_session_to_farmshare.sh 2>/dev/null || true
sleep 1
rm -f /tmp/farmshare-aws-session-push.pid
bash "$REPO/scripts/farmshare/_start_local_pusher.sh"
bash "$REPO/scripts/farmshare/push_aws_session_to_farmshare.sh" "$RUN"

rsync -avz -e "ssh -S $SOCK -o BatchMode=yes" \
  "$REPO/datasets/olmohq/publish_olmohq_edullm_data.py" \
  "$REPO/datasets/olmohq/publish_olmohq_skip_stage.sbatch" \
  "$HOST:$STAGING/datasets/olmohq/"

ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<EOF
set -Eeuo pipefail
RUN=$RUN
STAGING=$STAGING
echo "remote_host=\$(hostname)"
echo "remote_user=\$(whoami)"
# Cancel pending 128G job
scancel 1670742 || true
sleep 1
squeue -j 1670742 -o '%.18i %.2t %.10M %R' 2>/dev/null || echo "1670742 gone"
touch "\$RUN/STOP_AWS_REFRESH"
pkill -f "refresh_aws_session_loop.sh" -u nzhao2 2>/dev/null || true
mkdir -p "\$RUN/scripts/olmohq" "\$RUN/logs"
cp -a "\$STAGING/datasets/olmohq/publish_olmohq_edullm_data.py" "\$RUN/scripts/olmohq/"
cp -a "\$STAGING/datasets/olmohq/publish_olmohq_skip_stage.sbatch" "\$RUN/scripts/olmohq/"
sed -i 's/\r\$//' "\$RUN/scripts/olmohq/"*.py "\$RUN/scripts/olmohq/"*.sbatch
grep -n 'mem=' "\$RUN/scripts/olmohq/publish_olmohq_skip_stage.sbatch"
test -d "\$RUN/publish-stage/tokens/dclm"
source "\$RUN/aws-session.env"
export PATH="\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}"
aws sts get-caller-identity --query Arn --output text
JOB=\$(sbatch --parsable --exclude=wheat-01 --chdir="\$RUN" \
  --export=ALL,RUN_DIR="\$RUN",STAGE_DIR="\$RUN/publish-stage",EDULLM_ROOT="\$RUN",SCRIPTS="\$RUN/scripts/olmohq",AWS_SESSION_ENV="\$RUN/aws-session.env",BUCKET=edullm-datasets,PREFIX=olmo100b/olmo-mix-1124-30b \
  "\$RUN/scripts/olmohq/publish_olmohq_skip_stage.sbatch")
echo "skip_stage_127_job=\$JOB"
squeue -j "\$JOB" -o '%.18i %.9P %.22j %.2t %.10M %R %m'
EOF
