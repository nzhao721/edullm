#!/usr/bin/env bash
set -euo pipefail
RUN=/scratch/users/nzhao2/agent-runs/smollm2-135m-750m-27ep-20260730-162021
CKPT="${RUN}/output/checkpoints/step0011444"
LOG="${RUN}/logs/train-1670462.out"
while true; do
  if [[ -d "${CKPT}" && -f "${CKPT}/trainer_state.pt" && -f "${CKPT}/model.safetensors" ]]; then
    echo "READY $(date -Is)"
    ls "${RUN}/output/checkpoints/"
    tail -5 "${LOG}" || true
    exit 0
  fi
  if [[ -f "${LOG}" ]]; then
    line="$(tail -1 "${LOG}" || true)"
    echo "$(date +%H:%M:%S) waiting :: ${line}"
  else
    echo "$(date +%H:%M:%S) waiting :: no log"
  fi
  sleep 45
done
