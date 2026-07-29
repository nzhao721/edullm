#!/usr/bin/env bash
set -euo pipefail
for v in \
  /scratch/users/nzhao2/agent-runs/olmo-ladder-370m-20260722-185217/venv/bin/python \
  /scratch/users/nzhao2/agent-runs/rho-excess-10b-l40s/venv/bin/python
do
  echo "== $v =="
  if [[ -x "$v" ]]; then
    "$v" -c "import olmo_core; print('ok', olmo_core.__file__)" || echo FAIL
  else
    echo missing
  fi
done
