while ($true) {
  Start-Sleep -Seconds 60
  Write-Output 'AGENT_LOOP_TICK_mixlaw {"prompt":"Poll MixLaw pod es73etvp8x7zrq (mixlaw-370m-mix01-8xa100) via RunPod MCP stream-pod-logs. Report: phase (bootstrap/train/eval), step N/2384, tok/s, durable step if visible, disk/GPU health, errors. ALWAYS compute and report ETA: run `python scripts/runpod/mixlaw_poll_eta.py` on the latest step= log line (or pass --step/--tok-s from logs). Include ETA train-only, eval-buffer for remaining milestones (2125, 2384), ETA total with finish UTC, and ETA to next milestone. Intervene only on fatal issues (ENOSPC, crash, stuck eval >30m). Continue every 1 minute until step 2384 or failure."}'
}
