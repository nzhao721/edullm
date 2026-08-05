# Detached 5m poller for middle-PPL RunPod training. Appends tick lines for Cursor watcher.
$ErrorActionPreference = "Continue"
$log = Join-Path $env:TEMP "middle-ppl-monitor-ticks.txt"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$known = Join-Path $env:TEMP "runpod_known_hosts_vu6arqkxs0gv9h"
$remote = "root@185.216.23.188"
$port = 33884

function Write-Tick {
    param([string]$Body)
    $line = "AGENT_LOOP_TICK_middle_ppl $Body"
    Add-Content -Path $log -Value $line -Encoding utf8
}

function Fetch-Status {
    $cmd = "grep -E 'step=[0-9]+/|throughput/device/TPS=|eta=' /workspace/middle-ppl-optimized-train.log | tail -4"
    $raw = & ssh -i $key -p $port -o ConnectTimeout=15 -o StrictHostKeyChecking=no `
        -o UserKnownHostsFile=$known -o BatchMode=yes $remote $cmd 2>&1
    if ($LASTEXITCODE -ne 0) {
        return @{ error = ($raw -join "`n") }
    }
    $step = $null
    $eta = $null
    $tps = $null
    foreach ($line in $raw) {
        if ($line -match 'step=(\d+)/(\d+)') { $step = "$($Matches[1])/$($Matches[2])" }
        if ($line -match 'eta=([^,\]]+)') { $eta = $Matches[1].Trim() }
        if ($line -match 'throughput/device/TPS=([\d,]+)') { $tps = $Matches[1] }
    }
    return @{ step = $step; eta = $eta; tps_per_gpu = $tps }
}

if (-not (Test-Path $log)) {
    New-Item -Path $log -ItemType File -Force | Out-Null
}

while ($true) {
    try {
        $status = Fetch-Status
        $json = @{ prompt = "Monitor middle-PPL RunPod vu6arqkxs0gv9h: training step, throughput, ETA, errors. Be concise." }
        foreach ($k in $status.Keys) { $json[$k] = $status[$k] }
        Write-Tick ($json | ConvertTo-Json -Compress)
    }
    catch {
        Write-Tick (@{ prompt = "Monitor middle-PPL RunPod vu6arqkxs0gv9h"; error = $_.Exception.Message } | ConvertTo-Json -Compress)
    }
    Start-Sleep -Seconds 300
}
