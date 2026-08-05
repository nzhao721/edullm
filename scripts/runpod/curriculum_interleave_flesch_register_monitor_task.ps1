$script = "C:\alpha_ai\edullm\scripts\runpod\curriculum_interleave_flesch_poll_once.ps1"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File $script"
$start = (Get-Date).AddMinutes(1)
$trigger = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName 'CurriculumInterleaveFleschMonitorTick' -Action $action -Trigger $trigger -Settings $settings -Force
Get-ScheduledTask -TaskName 'CurriculumInterleaveFleschMonitorTick' | Select-Object TaskName, State
