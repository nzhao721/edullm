# 5m poller for curriculum interleave-flesch on RunPod. Emits AGENT_LOOP_TICK lines for Cursor.
$ErrorActionPreference = "Continue"
$tickLog = Join-Path $env:TEMP "curriculum-interleave-flesch-monitor-ticks.txt"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$known = Join-Path $env:TEMP "runpod_known_hosts_9g7tjgi3jrhn80"
$remote = "root@216.249.100.66"
$port = 23172
$podId = "9g7tjgi3jrhn80"
$trainLog = "/workspace/curriculum-interleave-flesch-train.log"
$emaLog = "/workspace/curriculum-interleave-flesch-ema.log"
$runDir = "/workspace/edullm-runs/curriculum/interleave-flesch"
$finalTaskLoss = "$runDir/progress/task_loss_results/step2384_task_loss.json"
$emaTaskLoss = "$runDir/progress/task_loss_results/step2385_task_loss.json"
$emaLegacyTaskLoss = "$runDir/checkpoints/step2384-ema/step2384-ema_task_loss.json"
$emaDoneMarker = "$runDir/progress/ema_integrated.done"
$emaLegacyDoneMarker = "$runDir/progress/ema_post_train.done"
$emaLock = "$runDir/progress/ema_post_train.lock"

function Write-Tick {
    param([string]$Body)
    $line = "AGENT_LOOP_TICK_CURRICULUM_INTERLEAVE_FLESCH $Body"
    Add-Content -Path $tickLog -Value $line -Encoding utf8
    Write-Output $line
}

function Fetch-Status {
    $cmd = @"
set +e
grep -E 'step=[0-9]+/2384|train/CE loss|train/PPL|throughput/total tokens|Training complete|EMA|step2385' $trainLog | tail -n 16
echo '---'
pgrep -c -f '/workspace/OLMo-core/.edullm/runpod/entrypoint.py' || echo 0
echo '---'
pgrep -f '[e]ntrypoint.py' >/dev/null && echo running || echo stopped
test -f $finalTaskLoss && echo eval_ok
grep -q 'Training complete' $trainLog && echo train_done
test -f $emaTaskLoss && echo ema_eval_ok
test -f $emaLegacyTaskLoss && echo ema_eval_ok
test -f $emaDoneMarker && echo ema_done
test -f $emaLegacyDoneMarker && echo ema_done
test -f $emaLock && echo ema_running
pgrep -f '[c]urriculum_ema.py' >/dev/null && echo ema_running
tail -n 3 $emaLog 2>/dev/null
exit 0
"@
    $raw = & ssh -i $key -p $port -o ConnectTimeout=15 -o StrictHostKeyChecking=no `
        -o UserKnownHostsFile=$known -o BatchMode=yes $remote $cmd 2>&1
    if ($LASTEXITCODE -ne 0) {
        return @{ error = ($raw -join "`n") }
    }
    $step = $null
    $stepNum = 0
    $eta = $null
    $tokens = $null
    $ce = $null
    $ppl = $null
    $procs = $null
    $trainerRunning = $false
    $evalOk = $false
    $trainDone = $false
    $emaEvalOk = $false
    $emaDone = $false
    $emaRunning = $false
    $emaLogTail = @()
    foreach ($line in $raw) {
        if ($line -match 'Training complete') { $trainDone = $true }
        if ($line -match 'step=(\d+)/(\d+)') {
            $stepNum = [int]$Matches[1]
            $step = "$($Matches[1])/$($Matches[2])"
        }
        if ($line -match 'eta=([^,\]]+)') { $eta = $Matches[1].Trim() }
        if ($line -match 'throughput/total tokens=([\d.]+[BMK]?)') { $tokens = $Matches[1] }
        if ($line -match 'train/CE loss=([\d.]+)') { $ce = $Matches[1] }
        if ($line -match 'train/PPL=([\d.]+)') { $ppl = $Matches[1] }
        if ($line -match '^(\d+)$') { $procs = $Matches[1] }
        if ($line -eq 'running') { $trainerRunning = $true }
        if ($line -eq 'eval_ok') { $evalOk = $true }
        if ($line -eq 'train_done') { $trainDone = $true }
        if ($line -eq 'ema_eval_ok') { $emaEvalOk = $true }
        if ($line -eq 'ema_done') { $emaDone = $true }
        if ($line -eq 'ema_running') { $emaRunning = $true }
        if ($line -match '^\d{4}-\d{2}-\d{2}' -or $line -match 'INFO|ERROR|macro_mean|EMA post-train') {
            $emaLogTail += $line
        }
    }
    $trainingComplete = (-not $trainerRunning) -and ($evalOk -or ($trainDone -and $stepNum -ge 2384))
    $emaComplete = $emaEvalOk -or $emaDone
    return @{
        step = $step
        step_num = $stepNum
        eta = $eta
        tokens = $tokens
        ce_loss = $ce
        ppl = $ppl
        trainer_procs = $procs
        trainer_running = $trainerRunning
        eval_ok = $evalOk
        train_done = $trainDone
        ema_eval_ok = $emaEvalOk
        ema_done = $emaDone
        ema_running = $emaRunning
        ema_log_tail = ($emaLogTail -join " | ")
        training_complete = $trainingComplete
        ema_complete = $emaComplete
        complete = $trainingComplete -and $emaComplete
    }
}

if (-not (Test-Path $tickLog)) {
    New-Item -Path $tickLog -ItemType File -Force | Out-Null
}

while ($true) {
    try {
        $status = Fetch-Status
        $json = @{
            prompt = if ($status.complete) {
                "Curriculum interleave-flesch training and EMA eval are complete on pod $podId. Summarize final step-2384 + EMA step-2385 metrics from W&B project curriculum. Do NOT terminate the pod unless the user asks."
            } elseif ($status.training_complete -and $status.ema_running) {
                "Curriculum interleave-flesch training is complete on pod $podId. Integrated EMA + step-2385 task-loss eval is running. Report EMA progress from train log and $emaLog."
            } elseif ($status.training_complete -and -not $status.ema_complete) {
                "Curriculum interleave-flesch training is complete on pod $podId. Integrated EMA should be running or pending."
            } else {
                "Check curriculum interleave-flesch RunPod training on pod $podId. Fetch $trainLog, verify 8-GPU trainer is running, and reply with a concise progress + ETA summary for the user."
            }
        }
        foreach ($k in $status.Keys) { $json[$k] = $status[$k] }
        Write-Tick ($json | ConvertTo-Json -Compress)
        if ($status.complete) { break }
    }
    catch {
        Write-Tick (@{
            prompt = "Check curriculum interleave-flesch RunPod training progress on pod $podId."
            error = $_.Exception.Message
        } | ConvertTo-Json -Compress)
    }
    Start-Sleep -Seconds 300
}
