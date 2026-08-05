# Detached 5m poller for curriculum RunPod training. Appends tick lines for Cursor watcher.
$ErrorActionPreference = "Continue"
$log = Join-Path $env:TEMP "curriculum-train-monitor-ticks.txt"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$known = Join-Path $env:TEMP "runpod_known_hosts"
$remote = "root@154.54.102.49"
$port = 17866
$podId = "l4xa2jyf9az1br"

function Write-Tick {
    param([string]$Body)
    $line = "AGENT_LOOP_TICK_CURRICULUM_TRAIN $Body"
    Add-Content -Path $log -Value $line -Encoding utf8
}

function Fetch-Status {
    $cmd = @"
grep -oE 'step=[0-9]+/2384' /workspace/curriculum-train.log | tail -1;
tail -n 30 /workspace/curriculum-train.log | grep -E 'eta=|train/CE loss=|throughput/total tokens=' | tail -3;
ps aux | grep -E 'torch.distributed.run.*curriculum' | grep -v grep | head -1 || echo 'trainer not running'
"@
    $raw = & ssh -i $key -p $port -o ConnectTimeout=15 -o StrictHostKeyChecking=no `
        -o UserKnownHostsFile=$known -o BatchMode=yes $remote $cmd 2>&1
    if ($LASTEXITCODE -ne 0) {
        return @{ error = ($raw -join "`n") }
    }
    $step = $null
    $eta = $null
    $loss = $null
    $tokens = $null
    $trainer = "unknown"
    foreach ($line in $raw) {
        if ($line -match 'step=(\d+)/2384') { $step = "$($Matches[1])/2384" }
        if ($line -match 'eta=([^,\]]+)') { $eta = $Matches[1].Trim() }
        if ($line -match 'train/CE loss=([\d.]+)') { $loss = $Matches[1] }
        if ($line -match 'throughput/total tokens=([\d.]+B?)') { $tokens = $Matches[1] }
        if ($line -match 'torch.distributed.run') { $trainer = "running" }
        if ($line -eq 'trainer not running') { $trainer = "not running" }
    }
    return @{
        pod = $podId
        step = $step
        eta = $eta
        ce_loss = $loss
        tokens = $tokens
        trainer = $trainer
    }
}

if (-not (Test-Path $log)) {
    New-Item -Path $log -ItemType File -Force | Out-Null
}

while ($true) {
    try {
        $status = Fetch-Status
        $json = @{
            prompt = "Check curriculum linear10-flesch RunPod training progress on pod $podId. Fetch /workspace/curriculum-train.log, verify 8-GPU trainer is still running, and reply with a concise progress + ETA summary for the user."
        }
        foreach ($k in $status.Keys) { $json[$k] = $status[$k] }
        Write-Tick ($json | ConvertTo-Json -Compress)
    }
    catch {
        Write-Tick (@{
            prompt = "Check curriculum linear10-flesch RunPod training progress on pod $podId"
            error = $_.Exception.Message
        } | ConvertTo-Json -Compress)
    }
    Start-Sleep -Seconds 300
}
