#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
RUN=/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z
JOB=1670744
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<EOF
set -e
echo "=== job ==="
sacct -j $JOB --format=JobID,State,ExitCode,Elapsed,NodeList,ReqMem,MaxRSS -P | head -8
sstat -j ${JOB}.batch --format=JobID,MaxRSS,AveRSS,AveCPU 2>/dev/null || true
echo "=== log stat ==="
stat -c '%y %s %n' $RUN/logs/olmo127b_edullm_publish_${JOB}.out $RUN/logs/olmo127b_edullm_publish_${JOB}.e
echo "=== out ==="
cat $RUN/logs/olmo127b_edullm_publish_${JOB}.out
echo "=== err tail ==="
tr -d '\\000' < $RUN/logs/olmo127b_edullm_publish_${JOB}.e | tail -40
echo "=== session ==="
stat -c 'mtime=%y' $RUN/aws-session.env
python3 -c "
from pathlib import Path
p=Path('$RUN/aws-session.env')
for line in p.read_text().splitlines():
    if line.startswith('export AWS_ACCESS_KEY_ID='):
        print('key_suffix=...'+line.split('=',1)[1][-4:])
    if 'EXPIR' in line.upper():
        print(line)
"
echo "=== publish py hints ==="
grep -nE 'def publish|upload|hash_worker|landing|print\\(' \\
  $RUN/edullm-data/src/edullm_data/*.py 2>/dev/null | head -50 || \\
grep -rnE 'def publish' $RUN/edullm-data/src/edullm_data --include='*.py' | head -20
EOF
