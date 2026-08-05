# 2m poller for warmup-flesch post-hoc EMA until complete.
$ErrorActionPreference = "Continue"
$tickLog = Join-Path $env:TEMP "curriculum-warmup-flesch-monitor-ticks.txt"
$pollScript = Join-Path $PSScriptRoot "curriculum_warmup_flesch_poll_once.ps1"

while ($true) {
    try {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $pollScript | Out-Null
        $last = Get-Content $tickLog -Tail 1 -ErrorAction SilentlyContinue
        if ($last -match '"complete":true') { break }
    }
    catch {
        $line = "AGENT_LOOP_TICK_CURRICULUM_WARMUP_FLESCH $(@{ error = $_.Exception.Message } | ConvertTo-Json -Compress)"
        Add-Content -Path $tickLog -Value $line -Encoding utf8
        Write-Output $line
    }
    Start-Sleep -Seconds 120
}
