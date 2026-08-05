while ($true) {
  Start-Sleep -Seconds 300
  Write-Output 'AGENT_LOOP_TICK_mixlaw_all {"prompt":"Run scripts/runpod/mixlaw_poll_all.ps1. It greps live [step=N/2384] from logs and auto-terminates any pod that is stopped + has step2384 eval (no idling). Report step table; note any TERMINATED. Under 6 lines."}'
}
