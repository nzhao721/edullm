# Arms Cursor-monitored tick watcher (pairs with detached monitor_daemon writing ticks every 5m).
$ErrorActionPreference = "Continue"
$lock = Join-Path $env:TEMP "curriculum-interleave-flesch-watch.lock"
# Kill orphan monitor_watch processes (not arm_watch — avoids killing the launching shell).
$myPid = $PID
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object {
    $_.ProcessId -ne $myPid -and $_.CommandLine -match 'curriculum_interleave_flesch_monitor_watch'
} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Remove-Item $lock -Force -ErrorAction SilentlyContinue
New-Item -Path $lock -ItemType File -Force | Out-Null
try {
    & "$PSScriptRoot\curriculum_interleave_flesch_monitor_watch.ps1"
}
finally {
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
}
