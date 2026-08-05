# Tail agent-loop tick log for curriculum learn capacity retries.
# The daemon is _curriculum_learn_agent_loop_tick.ps1; this echoes new ticks to stdout.
$log = Join-Path $env:TEMP "curriculum-learn-capacity-ticks.txt"
if (-not (Test-Path $log)) {
    New-Item -Path $log -ItemType File -Force | Out-Null
}
$pos = (Get-Item $log).Length
while ($true) {
    if (Test-Path $log) {
        $len = (Get-Item $log).Length
        if ($len -gt $pos) {
            $stream = [System.IO.File]::Open($log, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
            $stream.Position = $pos
            $reader = New-Object System.IO.StreamReader($stream)
            while (-not $reader.EndOfStream) {
                $line = $reader.ReadLine()
                if ($line -and $line.Trim().StartsWith('AGENT_LOOP_TICK_CURRICULUM_LEARN_CAPACITY')) {
                    Write-Output $line.Trim()
                }
            }
            $pos = $stream.Length
            $reader.Close()
            $stream.Close()
        }
    }
    Start-Sleep -Seconds 60
}
