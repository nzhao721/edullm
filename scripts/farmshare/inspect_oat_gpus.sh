#!/usr/bin/env bash
set -euo pipefail
for n in oat-01 oat-02 oat-03 oat-04 oat-05 oat-06; do
  echo "==== ${n}"
  scontrol show node "${n}" | tr ' ' '\n' | egrep 'NodeName=|State=|CfgTRES=|AllocTRES=|Gres=|GresUsed=' || true
done
echo "==== gpu queue"
squeue -p gpu -o '%.18i %.9P %.12j %.8u %.2t %.10M %.4D %R %b'
