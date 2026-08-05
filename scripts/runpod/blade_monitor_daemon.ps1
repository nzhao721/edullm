# 5m poller for BLADE instruct-v3 on RunPod. Emits AGENT_LOOP_TICK lines for Cursor.
$ErrorActionPreference = "Continue"
$tickLog = Join-Path $env:TEMP "blade-monitor-ticks.txt"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$known = Join-Path $env:TEMP "runpod_known_hosts_vu6arqkxs0gv9h"
$remote = "root@185.216.23.188"
$port = 33884
$podId = "vu6arqkxs0gv9h"
$trainLog = "/workspace/blade-train.log"
$runDir = "/workspace/edullm-runs/token-selection/blade"
$totalSteps = 2361
$firstSync = 500

function Write-Tick {
    param([string]$Body)
    $line = "AGENT_LOOP_TICK_BLADE_PROGRESS $Body"
    Add-Content -Path $tickLog -Value $line -Encoding utf8
    Write-Output $line
}

function Fetch-Status {
    $cmd = @"
set +e
grep -E 'step=[0-9]+/$totalSteps|throughput/total tokens|throughput/device/TPS|train/CE loss|Training complete|Traceback|OutOfMemory|ERROR' $trainLog | tail -n 16
echo '---'
pgrep -f '[e]ntrypoint.py --arm blade' >/dev/null && echo running || echo stopped
ls -1d $runDir/checkpoints/step* 2>/dev/null | tail -3
echo '---'
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -8
exit 0
"@
    $raw = & ssh -i $key -p $port -o ConnectTimeout=20 -o StrictHostKeyChecking=no `
        -o UserKnownHostsFile=$known -o BatchMode=yes $remote $cmd 2>&1
    if ($LASTEXITCODE -ne 0) {
        return @{ error = ($raw -join "`n") }
    }
    $step = $null
    $stepNum = 0
    $eta = $null
    $tokens = $null
    $tps = $null
    $ce = $null
    $trainerRunning = $false
    $trainDone = $false
    $errors = @()
    $checkpoints = @()
    foreach ($line in $raw) {
        if ($line -match 'Training complete') { $trainDone = $true }
        if ($line -match 'step=(\d+)/(\d+)') {
            $stepNum = [int]$Matches[1]
            $step = "$($Matches[1])/$($Matches[2])"
        }
        if ($line -match 'eta=([^,\]]+)') { $eta = $Matches[1].Trim() }
        if ($line -match 'throughput/total tokens=([\d.]+[BMK]?)') { $tokens = $Matches[1] }
        if ($line -match 'throughput/device/TPS \(actual avg\)=([\d,]+)') { $tps = $Matches[1] }
        if ($line -match 'train/CE loss=([\d.]+)') { $ce = $Matches[1] }
        if ($line -eq 'running') { $trainerRunning = $true }
        if ($line -match 'step\d+') { $checkpoints += $line.Trim() }
        if ($line -match 'Traceback|OutOfMemory|ERROR') { $errors += $line.Trim() }
    }
    $complete = $trainDone -or ((-not $trainerRunning) -and $stepNum -ge $totalSteps)
    return @{
        step = $step
        step_num = $stepNum
        eta_logged = $eta
        throughput_tokens = $tokens
        tps_per_gpu = $tps
        ce_loss = $ce
        trainer_running = $trainerRunning
        train_done = $trainDone
        checkpoints = ($checkpoints -join "; ")
        errors = ($errors -join "; ")
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
                "BLADE instruct-v3 training finished or stopped on pod $podId. Summarize final step, metrics, durable checkpoints, and W&B status. Do NOT terminate the pod unless the user asks."
            } else {
                "Check BLADE instruct-v3 RunPod training on pod $podId using $trainLog and $runDir. Report concise step/$totalSteps, health, sustained throughput, GPU memory/utilization, durable checkpoints, logged ETA, and adjusted ETA including BLADE sync/eval overhead."
            }
        }
        foreach ($k in $status.Keys) { $json[$k] = $status[$k] }
        Write-Tick ($json | ConvertTo-Json -Compress)
        if ($status.complete) { break }
    }
    catch {
        Write-Tick (@{
            prompt = "Check BLADE instruct-v3 RunPod training progress on pod $podId."
            error = $_.Exception.Message
        } | ConvertTo-Json -Compress)
    }
    Start-Sleep -Seconds 300
}
