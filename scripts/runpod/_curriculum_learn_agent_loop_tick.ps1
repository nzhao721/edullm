# Agent loop: wake chat every 2 minutes to retry curriculum linear10-learn capacity.
while ($true) {
  Start-Sleep -Seconds 120
  Write-Output 'AGENT_LOOP_TICK_CURRICULUM_LEARN_CAPACITY {"prompt":"Run scripts/runpod/launch_curriculum_linear10_learn_runpod_tick.ps1 for curriculum linear10-learn (arm 2, 8xA100). Exit codes: 0=training started (kill this capacity loop and report success), 2=blocked host 154.54.102.51 or 154.54.102.44 — pod was deleted before bootstrap (report and keep loop), 3=no 8xA100 capacity on any tier (report and keep loop). NEVER bootstrap or train on SSH hosts 154.54.102.51 or 154.54.102.44. The latter produced repeated pod loss and confirmed FUSE checkpoint I/O errors. Report concisely in chat: result, pod id + SSH host if any, cloud/GPU tier tried, whether blocked host was rejected, bootstrap/staging/training phase, W&B URL + step + ETA if training started. Under 8 lines unless error."}'
}
