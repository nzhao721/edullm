# Poll tick log every 60s and echo new lines to wake Cursor (pairs with detached monitor_daemon).
$log = Join-Path $env:TEMP "curriculum-interleave-flesch-monitor-ticks.txt"
if (-not (Test-Path $log)) {
    New-Item -Path $log -ItemType File -Force | Out-Null
}
$pos = (Get-Item $log).Length
$lastWakeStep = -1
while ($true) {
    if (Test-Path $log) {
        $len = (Get-Item $log).Length
        if ($len -gt $pos) {
            $stream = [System.IO.File]::Open($log, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
            $stream.Position = $pos
            $reader = New-Object System.IO.StreamReader($stream)
            while (-not $reader.EndOfStream) {
                $line = $reader.ReadLine()
                if ($line -and $line.Trim().StartsWith('AGENT_LOOP_TICK_CURRICULUM_INTERLEAVE_FLESCH')) {
                    if ($line -match 'AGENT_LOOP_TICK_CURRICULUM_INTERLEAVE_FLESCH (.+)') {
                        $status = $Matches[1] | ConvertFrom-Json
                        $stepNum = [int]$status.step_num
                        if ($stepNum -le $lastWakeStep) { continue }
                        $lastWakeStep = $stepNum
                        $prompt = if ($status.complete) {
                            "Curriculum interleave-flesch COMPLETE on pod 9g7tjgi3jrhn80. Summarize final metrics from W&B. Do NOT re-arm watch."
                        } else {
                            "Report interleave-flesch progress briefly (step, ETA, CE, PPL). Then re-arm watch: background Shell block_until_ms 0, notify_on_output pattern AGENT_LOOP_WAKE_CURRICULUM_INTERLEAVE_FLESCH, command: powershell -NoProfile -ExecutionPolicy Bypass -File C:\alpha_ai\edullm\scripts\runpod\curriculum_interleave_flesch_arm_watch.ps1"
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
                            complete = $status.complete
                        }
                        Write-Output "AGENT_LOOP_WAKE_CURRICULUM_INTERLEAVE_FLESCH $($wake | ConvertTo-Json -Compress)"
                    }
                }
            }
            $pos = $stream.Length
            $reader.Close()
            $stream.Close()
        }
    }
    Start-Sleep -Seconds 60
}
