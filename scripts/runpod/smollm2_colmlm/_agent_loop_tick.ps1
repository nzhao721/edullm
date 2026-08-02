while ($true) {
  Write-Output 'AGENT_LOOP_TICK_colmlm8 {"prompt":"Poll the 8xL40S Co-LMLM W&B-resume run on pod y83pcj0g00wijz via SSH. Use create_idle_pod.js --inspect for current host/port, then bash /workspace/edullm-smollm2-colmlm/poll_run_status.sh (or poll_resume8.sh during bootstrap). Report phase, progress, throughput, GPU util, errors. ALWAYS include ETA_train, ETA_total (with eval buffer + finish UTC from poll script), and ETA_next_eval when available. Continue every 5 minutes until user stops or run fails/completes."}'
  Start-Sleep -Seconds 300
}
