#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
# Prefer clone from recent publish run if present
RUN=/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<EOF
set -e
ED=$RUN/edullm-data
if [[ ! -d \$ED/src/edullm_data ]]; then
  echo "no local clone"
  exit 0
fi
echo "=== profiles ==="
ls \$ED/src/edullm_data/profiles/
echo "=== registry available ==="
\$RUN/venv/bin/python -c "from edullm_data.profiles.registry import available; print(sorted(available()))"
echo "=== publish signature ==="
\$RUN/venv/bin/python -c "import inspect; from edullm_data.publish import publish; print(inspect.signature(publish))"
echo "=== registry _SHIPPED / multi group hints ==="
grep -nE 'text-corpus|profile|multi|groups|_SHIPPED|available' \$ED/src/edullm_data/profiles/registry.py | head -40
echo "=== CONTRIBUTING profile section ==="
grep -nE 'text|multi-group|profile dict|group' \$ED/CONTRIBUTING.md 2>/dev/null | head -30 || true
grep -nE 'text-corpus|pretrain-tokens|profile' \$ED/README.md 2>/dev/null | head -30 || true
echo "=== families/pretrain ==="
cat \$ED/families/pretrain.json 2>/dev/null || true
EOF
