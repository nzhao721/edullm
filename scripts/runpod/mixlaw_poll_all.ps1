# Grep latest [step=N/2384] from each MixLaw pod log; terminate finished idle pods.
$key = "$env:USERPROFILE\.ssh\runpod_ed25519"
$known = "$env:TEMP\runpod_known_hosts"
$deleteJs = Join-Path $PSScriptRoot "smollm2_colmlm\create_idle_pod.js"

$targets = @(
  @{ Label = "olmo-mix"; PodId = "w208c8973c6rg8"; Host = "216.249.100.66"; Port = 22949; Arm = "olmo-mix-1124"; Log = "/workspace/edullm-runs/mixlaw/olmo-mix-1124/launch.out" }
)

foreach ($t in $targets) {
  $line = ssh -i $key -p $t.Port -o StrictHostKeyChecking=no -o UserKnownHostsFile=$known -o ConnectTimeout=10 `
    "root@$($t.Host)" "tail -n 50 $($t.Log) | grep '\[step=' | tail -1"
  if ($line) { Write-Output ("{0,-8} {1}" -f $t.Label, $line.Trim()) }
  else { Write-Output ("{0,-8} no log line" -f $t.Label) }

  $status = ssh -i $key -p $t.Port -o StrictHostKeyChecking=no -o UserKnownHostsFile=$known -o ConnectTimeout=10 `
    "root@$($t.Host)" @"
pgrep -f '[e]ntrypoint.py' >/dev/null && echo running || echo stopped
test -f /workspace/edullm-runs/mixlaw/$($t.Arm)/eval-work/task-loss/step2384_task_loss.json && echo eval_ok
"@

  if ($status -match "stopped" -and $status -match "eval_ok") {
    node $deleteJs --delete $t.PodId
    Write-Output ("{0,-8} TERMINATED pod $($t.PodId)" -f $t.Label)
  }
}
