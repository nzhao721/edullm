$script = "C:\alpha_ai\edullm\scripts\runpod\curriculum_mtld_poll_once.ps1"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File $script"
$start = (Get-Date).AddMinutes(1)
$trigger = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName 'CurriculumMtldMonitorTick' -Action $action -Trigger $trigger -Settings $settings -Force
Get-ScheduledTask -TaskName 'CurriculumMtldMonitorTick' | Select-Object TaskName, State
