# 5m poller for curriculum linear10-learn on RunPod. Emits AGENT_LOOP_TICK lines for Cursor.
$ErrorActionPreference = "Continue"
$tickLog = Join-Path $env:TEMP "curriculum-linear10-learn-monitor-ticks.txt"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$known = Join-Path $env:TEMP "runpod_known_hosts_vu6arqkxs0gv9h"
$remote = "root@185.216.23.188"
$port = 33884
$podId = "vu6arqkxs0gv9h"
$trainLog = "/workspace/curriculum-linear10-learn-train.log"
$runDir = "/workspace/edullm-runs/curriculum/linear10-learn"
$finalTaskLoss = "$runDir/progress/task_loss_results/step2384_task_loss.json"

function Write-Tick {
    param([string]$Body)
    $line = "AGENT_LOOP_TICK_CURRICULUM_LINEAR10_LEARN $Body"
    Add-Content -Path $tickLog -Value $line -Encoding utf8
    Write-Output $line
}

function Fetch-Status {
    $cmd = @"
set +e
grep -E 'step=[0-9]+/2384|train/CE loss|train/PPL|throughput/total tokens|Training complete' $trainLog | tail -n 12
echo '---'
pgrep -c -f '/workspace/OLMo-core/.edullm/runpod/entrypoint.py' || echo 0
echo '---'
pgrep -f '[e]ntrypoint.py' >/dev/null && echo running || echo stopped
test -f $finalTaskLoss && echo eval_ok
grep -q 'Training complete' $trainLog && echo train_done
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
    }
    $complete = (-not $trainerRunning) -and ($evalOk -or ($trainDone -and $stepNum -ge 2384))
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
        complete = $complete
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
                "Curriculum linear10-learn training is complete on pod $podId (step 2384 eval done). Summarize final metrics. Do NOT terminate the pod unless the user asks."
            } else {
                "Check curriculum linear10-learn RunPod training on pod $podId. Fetch $trainLog, verify 8-GPU trainer is running, and reply with a concise progress + ETA summary for the user."
            }
        }
        foreach ($k in $status.Keys) { $json[$k] = $status[$k] }
        Write-Tick ($json | ConvertTo-Json -Compress)
        if ($status.complete) { break }
    }
    catch {
        Write-Tick (@{
            prompt = "Check curriculum linear10-learn RunPod training progress on pod $podId."
            error = $_.Exception.Message
        } | ConvertTo-Json -Compress)
    }
    Start-Sleep -Seconds 300
}
