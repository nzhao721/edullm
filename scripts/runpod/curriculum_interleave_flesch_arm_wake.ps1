# Entry point for Cursor monitored background shell (one-shot 5m wake).
$ErrorActionPreference = "Continue"
$lock = Join-Path $env:TEMP "curriculum-interleave-flesch-wake-once.lock"
if (Test-Path $lock) {
    $age = (Get-Date) - (Get-Item $lock).LastWriteTime
    if ($age.TotalMinutes -lt 6) {
        Write-Output "AGENT_LOOP_WAKE_CURRICULUM_INTERLEAVE_FLESCH_ALREADY_ARMED"
        exit 0
    }
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
}
New-Item -Path $lock -ItemType File -Force | Out-Null
try {
    & "$PSScriptRoot\curriculum_interleave_flesch_wake_once.ps1"
}
finally {
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
}
