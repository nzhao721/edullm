# One-shot curriculum linear10-mtld status poll. Used by scheduled task and manual runs.
# When training completes: run post-hoc EMA eval into the same W&B run (step 2385).
# Pod termination is disabled — leave the pod running until the user says otherwise.
$ErrorActionPreference = "Continue"
$lock = Join-Path $env:TEMP "curriculum-mtld-monitor-tick.lock"
if (Test-Path $lock) {
    $lockAge = (Get-Date) - (Get-Item $lock).LastWriteTime
    if ($lockAge.TotalMinutes -lt 4) { exit 0 }
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
}
New-Item -Path $lock -ItemType File -Force | Out-Null

$tickLog = Join-Path $env:TEMP "curriculum-mtld-monitor-ticks.txt"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$known = Join-Path $env:TEMP "runpod_known_hosts_vu6arqkxs0gv9h"
$remote = "root@185.216.23.188"
$port = 33884
$podId = "vu6arqkxs0gv9h"
$runDir = "/workspace/edullm-runs/curriculum/linear10-mtld"
$trainLog = "/workspace/curriculum-linear10-mtld-train.log"
$finalTaskLoss = "$runDir/progress/task_loss_results/step2384_task_loss.json"
$emaTaskLoss = "$runDir/progress/task_loss_results/step2385_task_loss.json"
$emaLegacyTaskLoss = "$runDir/checkpoints/step2384-ema/step2384-ema_task_loss.json"
$emaDoneMarker = "$runDir/progress/ema_integrated.done"
$emaLegacyDoneMarker = "$runDir/progress/ema_post_train.done"
$emaLock = "$runDir/progress/ema_post_train.lock"
$emaRemoteScript = "/workspace/curriculum_mtld_post_train_ema.sh"
$repoDir = "/workspace/OLMo-core-curriculum-mtld"
$olmoWorktree = "C:\alpha_ai\OLMo-core-curriculum-370m"
$postTrainScript = Join-Path $PSScriptRoot "curriculum_mtld_post_train_ema.sh"
$emaPy = Join-Path $olmoWorktree ".edullm\curriculum_ema.py"

function Write-Tick {
    param([string]$Body)
    $line = "AGENT_LOOP_TICK_CURRICULUM_MTLD_TRAIN $Body"
    Add-Content -Path $tickLog -Value $line -Encoding utf8
}

function Invoke-Ssh {
    param([string]$Cmd)
    & ssh -i $key -p $port -o ConnectTimeout=15 -o StrictHostKeyChecking=no `
        -o UserKnownHostsFile=$known -o BatchMode=yes $remote $Cmd 2>&1
}

function Invoke-Scp {
    param([string]$LocalPath, [string]$RemotePath)
    & scp -i $key -P $port -o ConnectTimeout=15 -o StrictHostKeyChecking=no `
        -o UserKnownHostsFile=$known $LocalPath "${remote}:${RemotePath}" 2>&1
}

function Sync-PostTrainAssets {
    if (-not (Test-Path $postTrainScript)) {
        throw "post-train script missing: $postTrainScript"
    }
    if (-not (Test-Path $emaPy)) {
        throw "curriculum_ema.py missing: $emaPy"
    }
    Invoke-Scp $postTrainScript $emaRemoteScript | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "failed to copy post-train script" }
    Invoke-Scp $emaPy "$repoDir/.edullm/curriculum_ema.py" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "failed to copy curriculum_ema.py" }
    Invoke-Ssh "chmod +x $emaRemoteScript" | Out-Null
}

function Fetch-Status {
    $cmd = @"
set +e
grep -E 'step=[0-9]+/2384|train/CE loss|train/PPL|throughput/total tokens|Training complete' $trainLog | tail -n 12
echo '---'
pgrep -c -f '/workspace/OLMo-core-curriculum-mtld/.edullm/runpod/entrypoint.py' || echo 0
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
exit 0
"@
    $raw = Invoke-Ssh $cmd
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
    }
    $trainingComplete = (-not $trainerRunning) -and ($evalOk -or ($trainDone -and $stepNum -ge 2384))
    $emaComplete = $emaEvalOk -or $emaDone
    $readyToTerminate = $trainingComplete -and $emaComplete
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
        training_complete = $trainingComplete
        ema_complete = $emaComplete
        complete = $readyToTerminate
    }
}

function Invoke-StartPostTrainEma {
    Sync-PostTrainAssets
    $start = Invoke-Ssh "nohup bash $emaRemoteScript >> /workspace/curriculum-linear10-mtld-ema.log 2>&1 & echo started"
    if ($LASTEXITCODE -ne 0) {
        throw "failed to start post-train EMA: $($start -join "`n")"
    }
    return @{ ema_started = $true; ema_output = ($start -join "`n").Trim() }
}

try {
    if (-not (Test-Path $tickLog)) {
        New-Item -Path $tickLog -ItemType File -Force | Out-Null
    }
    $status = Fetch-Status
    $emaAction = $null

    if ($status.training_complete -and -not $status.ema_complete -and -not $status.ema_running) {
        $emaAction = Invoke-StartPostTrainEma
        $status.ema_running = $true
    }

    $json = @{
        prompt = if ($status.complete) {
            "Curriculum linear10-mtld training and EMA eval are complete on pod $podId. Summarize final step/loss + EMA logged to W&B step 2385. Do NOT terminate the pod."
        } elseif ($status.training_complete -and ($status.ema_running -or $emaAction)) {
            "Curriculum linear10-mtld training is complete on pod $podId. Post-hoc EMA + step-2385 task-loss eval is running. Do NOT terminate the pod."
        } elseif ($status.training_complete) {
            "Curriculum linear10-mtld training is complete on pod $podId. Start post-hoc EMA eval (steps 2000/2125/2250/2384) into the same W&B run at step 2385. Do NOT terminate the pod."
        } else {
            "Check curriculum linear10-mtld RunPod training progress on pod $podId. Fetch $trainLog, verify the 8-GPU trainer is still running, and reply with a concise progress + ETA summary for the user."
        }
    }
    foreach ($k in $status.Keys) { $json[$k] = $status[$k] }
    if ($emaAction) {
        foreach ($k in $emaAction.Keys) { $json[$k] = $emaAction[$k] }
    }
    $json["terminate_pod"] = $false
    Write-Tick ($json | ConvertTo-Json -Compress)
}
catch {
    Write-Tick (@{
        prompt = "Check curriculum linear10-mtld RunPod training progress on pod $podId."
        error = $_.Exception.Message
    } | ConvertTo-Json -Compress)
}
finally {
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
}
