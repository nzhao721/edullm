# Smoke-test OLMo-core 370M on RunPod 8xA100 via real .edullm/runpod scripts.
param(
  [ValidateSet("mixlaw", "curriculum")]
  [string]$Experiment = "mixlaw",
  [string]$Profile = "sbsandbox",
  [string]$GpuType = "NVIDIA A100-SXM4-80GB",
  [int]$GpuCount = 8,
  [int]$VolumeGb = 250,
  [int]$LengthTokens = 25165824,
  [int]$ArmIndex = 0,
  [string]$CloudType = "COMMUNITY",
  [int]$PollSeconds = 5
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$createPod = Join-Path $PSScriptRoot "smollm2_colmlm\create_idle_pod.js"
$mintScript = Join-Path $repoRoot "scripts\farmshare\mint_aws_session_local.ps1"
$wandbKeyFile = Join-Path $env:USERPROFILE ".wandb_api_key"
$sshKey = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$sshPub = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519.pub"
$awsEnv = Join-Path $env:TEMP "aws-session-runpod-smoke.env"
$wandbEnv = Join-Path $env:TEMP "wandb-session-runpod-smoke.env"
$logLocal = Join-Path $env:TEMP "olmo-core-runpod-smoke.log"

switch ($Experiment) {
  "mixlaw" {
    $OlmoCoreRoot = "C:\alpha_ai\OLMo-core"
    $podName = "mixlaw-370m-smoke-8xa100"
    $repoDir = "/workspace/olmo-core-mixlaw-smoke"
    $inputFamily = "mixlaw"
    $runFamily = "mixlaw"
  }
  "curriculum" {
    $OlmoCoreRoot = "C:\alpha_ai\OLMo-core-curriculum-370m"
    $podName = "curriculum-370m-smoke-8xa100"
    $repoDir = "/workspace/olmo-core-curriculum-smoke"
    $inputFamily = "curriculum"
    $runFamily = "curriculum"
  }
}

if (-not (Test-Path $OlmoCoreRoot)) { throw "OLMo-core worktree missing: $OlmoCoreRoot" }
if (-not (Test-Path (Join-Path $OlmoCoreRoot ".edullm\runpod\launch.sh"))) {
  throw "missing runpod adapter under $OlmoCoreRoot\.edullm\runpod"
}

function Invoke-Ssh {
  param([string]$HostName, [int]$Port, [string]$RemoteCommand)
  & ssh -i $sshKey -p $Port -o StrictHostKeyChecking=no -o UserKnownHostsFile=$env:TEMP\runpod_known_hosts -o BatchMode=yes "root@${HostName}" $RemoteCommand
  if ($LASTEXITCODE -ne 0) { throw "ssh failed: $RemoteCommand" }
}

function Invoke-Scp {
  param([string]$LocalPath, [string]$HostName, [int]$Port, [string]$RemotePath)
  & scp -i $sshKey -P $Port -o StrictHostKeyChecking=no -o UserKnownHostsFile=$env:TEMP\runpod_known_hosts $LocalPath "root@${HostName}:$RemotePath"
  if ($LASTEXITCODE -ne 0) { throw "scp failed: $LocalPath -> $RemotePath" }
}

Write-Host "Creating RunPod $GpuCount x $GpuType ($Experiment, $CloudType)..."
$createJson = node $createPod $GpuType $podName $GpuCount $VolumeGb $sshPub $CloudType
Write-Host $createJson
$podInfo = $createJson | ConvertFrom-Json
$podId = $podInfo.podId
$sshHost = $podInfo.sshHost
$sshPort = [int]$podInfo.sshPort
Write-Host "podId=$podId ssh=${sshHost}:$sshPort"

try {
  & $mintScript -Profile $Profile -OutputPath $awsEnv

  Write-Host "Preparing $repoDir..."
  Invoke-Ssh $sshHost $sshPort "rm -rf '$repoDir' /workspace/edullm-inputs/$inputFamily /workspace/edullm-runs/$runFamily"
  Invoke-Ssh $sshHost $sshPort "test ! -e '$repoDir' || (echo 'failed to clear $repoDir' >&2; exit 2)"

  $bootstrap = Join-Path $OlmoCoreRoot ".edullm\runpod\bootstrap.sh"
  Invoke-Scp $bootstrap $sshHost $sshPort "/tmp/bootstrap.sh"
  Invoke-Ssh $sshHost $sshPort "chmod +x /tmp/bootstrap.sh && REPO_DIR='$repoDir' bash /tmp/bootstrap.sh"

  Invoke-Ssh $sshHost $sshPort "mkdir -p '$repoDir/.edullm/runpod'"
  & scp -i $sshKey -P $sshPort -o StrictHostKeyChecking=no -o UserKnownHostsFile=$env:TEMP\runpod_known_hosts -r `
    (Join-Path $OlmoCoreRoot ".edullm\runpod\*") "root@${sshHost}:$repoDir/.edullm/runpod/"
  if ($LASTEXITCODE -ne 0) { throw "scp runpod adapter failed" }
  if ($Experiment -eq "mixlaw") {
    Invoke-Scp (Join-Path $OlmoCoreRoot ".edullm\mixlaw_entrypoint.py") `
      $sshHost $sshPort "$repoDir/.edullm/mixlaw_entrypoint.py"
  } else {
    Invoke-Scp (Join-Path $OlmoCoreRoot ".edullm\curriculum_entrypoint.py") `
      $sshHost $sshPort "$repoDir/.edullm/curriculum_entrypoint.py"
    Invoke-Scp (Join-Path $OlmoCoreRoot ".edullm\curriculum_recipe.json") `
      $sshHost $sshPort "$repoDir/.edullm/curriculum_recipe.json"
  }

  Invoke-Scp $awsEnv $sshHost $sshPort "/workspace/aws-session.env"
  Invoke-Ssh $sshHost $sshPort "chmod 600 /workspace/aws-session.env"
  Remove-Item -Force $awsEnv -ErrorAction SilentlyContinue
  if ($Experiment -eq "mixlaw") {
    $stageCmd = "cd '$repoDir' && PYTHONPATH='$repoDir/src:$repoDir/.edullm' python3 .edullm/runpod/stage_inputs.py --credentials-file /workspace/aws-session.env --arm-index $ArmIndex --length-tokens $LengthTokens --max-files-per-domain 1"
  } else {
    $stageCmd = "cd '$repoDir' && PYTHONPATH='$repoDir/src:$repoDir/.edullm' python3 .edullm/runpod/stage_inputs.py --credentials-file /workspace/aws-session.env --arm-index $ArmIndex"
  }
  Write-Host "Staging inputs..."
  Invoke-Ssh $sshHost $sshPort $stageCmd

  $wandbKey = (Get-Content $wandbKeyFile -Raw).Trim()
  $wandbContent = "export WANDB_API_KEY='$wandbKey'`n" +
    "export WANDB_ENTITY='eduLLM'`n" +
    "export WANDB_START_METHOD=thread`n"
  [IO.File]::WriteAllText($wandbEnv, $wandbContent, [Text.Encoding]::ASCII)
  Invoke-Scp $wandbEnv $sshHost $sshPort "/workspace/wandb-session.env"
  Invoke-Ssh $sshHost $sshPort "chmod 600 /workspace/wandb-session.env"

  Write-Host "Launching LENGTH_TOKENS=$LengthTokens (6 steps)..."
  $launchCmd = "cd '$repoDir' && setsid -f env REPO_DIR='$repoDir' ARM_INDEX=$ArmIndex LENGTH_TOKENS=$LengthTokens PYTORCH_ALLOC_CONF=expandable_segments:True bash .edullm/runpod/launch.sh </dev/null > /workspace/smoke-train.log 2>&1; echo started"
  Invoke-Ssh $sshHost $sshPort $launchCmd

  $deadline = (Get-Date).AddMinutes(60)
  $seenSteps = @{}
  $throughput = @()
  while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds $PollSeconds
    & scp -i $sshKey -P $sshPort -o StrictHostKeyChecking=no -o UserKnownHostsFile=$env:TEMP\runpod_known_hosts `
      "root@${sshHost}:/workspace/smoke-train.log" $logLocal 2>$null
    if (-not (Test-Path $logLocal)) {
      Write-Host "poll waiting for log..."
      continue
    }
    $lines = Get-Content $logLocal -ErrorAction SilentlyContinue
    foreach ($line in $lines) {
      if ($line -match 'throughput/device/TPS(?: \(actual avg\))?=([0-9][0-9,]*(?:\.[0-9]+)?)') {
        $throughput += [double]($Matches[1] -replace ',', '')
      }
      if ($line -match '\[step=(\d+)/') { $seenSteps[$Matches[1]] = $true }
    }
    $stepCount = $seenSteps.Keys.Count
    $lastTps = if ($throughput.Count) { $throughput[-1] } else { 0 }
    Write-Host ("poll t=$(Get-Date -Format 'HH:mm:ss') steps=$stepCount tps=$lastTps bytes=" + (Get-Item $logLocal).Length)
    if ($stepCount -ge 6 -or ($throughput.Count -ge 1 -and $stepCount -ge 3)) {
      Write-Host "Smoke target reached; stopping training..."
      Invoke-Ssh $sshHost $sshPort "pkill -INT -f '[t]orch.distributed.run.*entrypoint.py' || true"
      break
    }
  }

  if (Test-Path $logLocal) {
    Write-Host "--- tail smoke-train.log ---"
    Get-Content $logLocal -Tail 50
    Select-String -Path $logLocal -Pattern 'throughput|TPS|tokens/s' | Select-Object -Last 15 | ForEach-Object { $_.Line }
  }
}
finally {
  if ($podId) {
    Write-Host "Terminating pod $podId..."
    node $createPod --delete $podId
  }
  Remove-Item -Force $awsEnv, $wandbEnv -ErrorAction SilentlyContinue
}
