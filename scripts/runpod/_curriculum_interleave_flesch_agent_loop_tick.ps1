# Agent loop: wake chat every 5 minutes to poll interleave-flesch training.
while ($true) {
    Start-Sleep -Seconds 300
    $line = & "$PSScriptRoot\curriculum_interleave_flesch_poll_once.ps1" | Select-Object -Last 1
    if ($line) { Write-Output $line }
}
