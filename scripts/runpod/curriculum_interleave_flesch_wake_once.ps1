# Sleep 5m, poll training, emit AGENT_LOOP_WAKE for Cursor chat (one-shot; re-arm after each report).
$ErrorActionPreference = "Continue"
Start-Sleep -Seconds 300

$line = & "$PSScriptRoot\curriculum_interleave_flesch_poll_once.ps1" | Select-Object -Last 1
if (-not $line) { exit 0 }

if ($line -match 'AGENT_LOOP_TICK_CURRICULUM_INTERLEAVE_FLESCH (.+)') {
    $status = $Matches[1] | ConvertFrom-Json
    $prompt = if ($status.complete) {
        "Curriculum interleave-flesch training and EMA are COMPLETE on pod 9g7tjgi3jrhn80. Summarize step-2384 + EMA step-2385 from W&B project curriculum. Do NOT re-arm the 5m wake loop. Do NOT terminate the pod unless the user asks."
    } else {
        "Report interleave-flesch progress briefly (step, ETA, CE loss, PPL, trainer status) using the attached status fields. Then re-arm watch: background Shell block_until_ms 0, notify_on_output pattern AGENT_LOOP_WAKE_CURRICULUM_INTERLEAVE_FLESCH, command: powershell -NoProfile -ExecutionPolicy Bypass -File C:\alpha_ai\edullm\scripts\runpod\curriculum_interleave_flesch_arm_watch.ps1"
    }
    $wake = @{
        prompt = $prompt
        step = $status.step
        step_num = $status.step_num
        eta = $status.eta
        ce_loss = $status.ce_loss
        ppl = $status.ppl
        tokens = $status.tokens
        trainer_running = $status.trainer_running
        training_complete = $status.training_complete
        ema_complete = $status.ema_complete
        complete = $status.complete
    }
    Write-Output "AGENT_LOOP_WAKE_CURRICULUM_INTERLEAVE_FLESCH $($wake | ConvertTo-Json -Compress)"
}
