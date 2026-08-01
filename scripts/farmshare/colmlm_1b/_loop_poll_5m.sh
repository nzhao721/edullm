#!/usr/bin/env bash
# 5-minute FarmShare progress poll loop for Co-LMLM 1B corpus.
while true; do
  sleep 300
  echo 'AGENT_LOOP_TICK_colmlm1b {"prompt":"Poll Co-LMLM 1B FarmShare corpus assembly progress via SSH status script; report job states, ingest percent, docs/spans/mark/qa status, data sizes, and ETA."}'
done
