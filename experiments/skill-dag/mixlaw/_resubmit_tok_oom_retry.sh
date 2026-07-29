#!/usr/bin/env bash
# Resubmit OOM'd pes2o tokenize tasks with 64G; then topup upload afterok.
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu

ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
TOP=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"
unset PREFIX || true

# Exact failed / incomplete indices from original array 1666301.
python3 - <<'PY'
from pathlib import Path
top = Path("/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841")
lines = (top/"tokenize_map.txt").read_text().splitlines()
missing = []
for i, line in enumerate(lines):
    dst = Path(line.split("|",1)[1])
    if not dst.is_file():
        missing.append(i)
        # clean stale tmp
        for p in dst.parent.glob(dst.name + "*"):
            if p.suffix == ".tmp" or str(p).endswith(".npy.tmp"):
                try:
                    p.unlink()
                    print(f"removed {p}")
                except Exception as e:
                    print(f"keep {p}: {e}")
(top/"plan/tok_retry_indices.txt").write_text("\n".join(str(i) for i in missing) + "\n")
print(f"retry_indices={missing}")
PY

# Cancel old pending upload and any leftover running tok tasks from 1666301.
scancel 1666759 2>/dev/null || true
# Let any still-running original tasks finish or fail; don't scancel mid-write if helpful —
# but OOM leftovers should be cancelled so we don't double-write.
scancel 1666301 2>/dev/null || true

NRETRY=$(wc -l < "$TOP/plan/tok_retry_indices.txt" | tr -d ' ')
echo "NRETRY=$NRETRY"

# Mint fresh AWS session for upcoming upload.
export EDULLM_ROOT=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
export RUN_DIR="$TOP"
source "$TOP/scripts/prepare_aws_session_light.sh"
source "$AWS_SESSION_ENV"
aws sts get-caller-identity --output text
source "$TOP/env.sh" || true
cat > "$TOP/env.sh" <<EOF
RUN_DIR=${TOP}
VENV=${TOP}/venv
EDULLM_ROOT=${EDULLM_ROOT}
AWS_SESSION_ENV=${AWS_SESSION_ENV}
BUCKET=${BUCKET:-edullm-datasets}
OLMOHQ_PREFIX=${OLMOHQ_PREFIX:-olmo100b/olmo-mix-1124-30b}
N=${N:-413}
EOF

cat > "$TOP/scripts/tokenize_topup_retry.sbatch" <<'EOF'
#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=wheat-01
set -Eeuo pipefail
unset PREFIX || true
source "${RUN_DIR}/env.sh"
source "${VENV}/bin/activate"
IDX=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${RUN_DIR}/plan/tok_retry_indices.txt")
LINE=$(sed -n "$((IDX + 1))p" "${RUN_DIR}/tokenize_map.txt")
SRC="${LINE%%|*}"
DST="${LINE##*|}"
mkdir -p "$(dirname "${DST}")"
echo "retry index=${IDX} src=${SRC} dst=${DST}"
python "${RUN_DIR}/scripts/tokenize_olmo_shard.py" --input "${SRC}" --output "${DST}"
EOF
sed -i 's/\r$//' "$TOP/scripts/tokenize_topup_retry.sbatch"

TOK_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --array=0-$((NRETRY - 1))%7 \
  --chdir="$TOP" \
  --output="$TOP/logs/tok-retry-%A_%a.out" \
  --error="$TOP/logs/tok-retry-%A_%a.err" \
  --export=ALL,RUN_DIR="$TOP",VENV="$TOP/venv" \
  "$TOP/scripts/tokenize_topup_retry.sbatch")
echo "tok_retry_job_id=$TOK_JOB"
echo "$TOK_JOB" > "$TOP/tokenize_retry_job_id.txt"

UP=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal --cpus-per-task=8 --mem=32G --time=12:00:00 \
  --dependency="afterok:${TOK_JOB}" \
  --job-name=topup-up --chdir="$TOP" \
  --output="$TOP/logs/upload-%j.out" --error="$TOP/logs/upload-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; export PATH=\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}; source $TOP/env.sh; source \$AWS_SESSION_ENV; source $TOP/venv/bin/activate; command -v aws; aws sts get-caller-identity --output text; python $TOP/scripts/finalize_olmohq_topup_upload.py --run-dir $TOP --bucket \$BUCKET --prefix \$OLMOHQ_PREFIX'")
echo "topup_upload_job_id=$UP"
echo "$UP" > "$TOP/upload_job_id.txt"

squeue --me -o '%.12i %.10T %.22j' | head -25
REMOTE
