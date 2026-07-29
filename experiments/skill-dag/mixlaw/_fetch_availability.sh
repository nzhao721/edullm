#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" 'cat /scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841/plan/availability_after_topup.json'
