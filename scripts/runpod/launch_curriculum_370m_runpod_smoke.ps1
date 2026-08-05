# Smoke-test curriculum 370M on RunPod 8xA100 using real .edullm/runpod scripts.
# Stages inputs, runs launch.sh with LENGTH_TOKENS (6 steps), monitors throughput, stops pod.
param(
  [string]$Profile = "sbsandbox",
  [string]$GpuType = "NVIDIA A100-SXM4-80GB",
  [int]$GpuCount = 8,
  [int]$VolumeGb = 250,
  [int]$LengthTokens = 25165824,
  [int]$ArmIndex = 0,
  [string]$CloudType = "COMMUNITY",
  [string]$OlmoCoreRoot = "C:\alpha_ai\OLMo-core-curriculum-370m"
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
$logLocal = Join-Path $env:TEMP "curriculum-runpod-smoke.log"
$podName = "curriculum-370m-smoke-8xa100"

if (-not (Test-Path $OlmoCoreRoot)) {
  throw "OLMo-core worktree missing: $OlmoCoreRoot"
}
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

Write-Host "Creating RunPod $GpuCount x $GpuType ($CloudType)..."
$createJson = node $createPod $GpuType $podName $GpuCount $VolumeGb $sshPub $CloudType
Write-Host $createJson
$podInfo = $createJson | ConvertFrom-Json
$podId = $podInfo.podId
$sshHost = $podInfo.sshHost
$sshPort = [int]$podInfo.sshPort
Write-Host "podId=$podId ssh=${sshHost}:$sshPort"

try {
  Write-Host "Minting temporary AWS session (laptop)..."
  & $mintScript -Profile $Profile -OutputPath $awsEnv

$repoDir = "/workspace/OLMo-core-smoke"

  Write-Host "Preparing clean checkout at $repoDir..."
  Invoke-Ssh $sshHost $sshPort "rm -rf '$repoDir' /workspace/edullm-inputs/curriculum /workspace/edullm-runs/curriculum"

  Write-Host "Running bootstrap.sh on pod..."
  $bootstrap = Join-Path $OlmoCoreRoot ".edullm\runpod\bootstrap.sh"
  Invoke-Scp $bootstrap $sshHost $sshPort "/tmp/bootstrap.sh"
  Invoke-Ssh $sshHost $sshPort "chmod +x /tmp/bootstrap.sh && REPO_DIR='$repoDir' bash /tmp/bootstrap.sh"

  Write-Host "Copying local runpod adapter (not on remote branch)..."
  Invoke-Ssh $sshHost $sshPort "mkdir -p '$repoDir/.edullm/runpod'"
  & scp -i $sshKey -P $sshPort -o StrictHostKeyChecking=no -o UserKnownHostsFile=$env:TEMP\runpod_known_hosts -r `
    (Join-Path $OlmoCoreRoot ".edullm\runpod\*") "root@${sshHost}:$repoDir/.edullm/runpod/"
  if ($LASTEXITCODE -ne 0) { throw "scp runpod adapter failed" }

  Write-Host "Staging sealed inputs (arm $ArmIndex)..."
  Invoke-Scp $awsEnv $sshHost $sshPort "/workspace/aws-session.env"
  Invoke-Ssh $sshHost $sshPort "chmod 600 /workspace/aws-session.env"
  Invoke-Ssh $sshHost $sshPort @"
cd '$repoDir' && PYTHONPATH='$repoDir/src:$repoDir/.edullm' python3 .edullm/runpod/stage_inputs.py --credentials-file /workspace/aws-session.env --arm-index $ArmIndex
"@

  if (-not (Test-Path $wandbKeyFile)) { throw "missing W&B key file: $wandbKeyFile" }
  $wandbKey = (Get-Content $wandbKeyFile -Raw).Trim()
  @"
# Generated for RunPod smoke. Do not commit.
export WANDB_API_KEY='$wandbKey'
export WANDB_ENTITY='eduLLM'
export WANDB_START_METHOD=thread
"@ | Set-Content -NoNewline -Encoding ascii $wandbEnv
  Invoke-Scp $wandbEnv $sshHost $sshPort "/workspace/wandb-session.env"
  Invoke-Ssh $sshHost $sshPort "chmod 600 /workspace/wandb-session.env"

  Write-Host "Launching training (LENGTH_TOKENS=$LengthTokens, 6 steps)..."
  $launchCmd = "cd '$repoDir' && REPO_DIR='$repoDir' ARM_INDEX=$ArmIndex LENGTH_TOKENS=$LengthTokens nohup bash .edullm/runpod/launch.sh > /workspace/smoke-train.log 2>&1 & echo started"
  $startOut = Invoke-Ssh $sshHost $sshPort $launchCmd
  Write-Host $startOut

  $deadline = (Get-Date).AddMinutes(45)
  $seenSteps = @{}
  $throughput = @()
  while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 20
    & scp -i $sshKey -P $sshPort -o StrictHostKeyChecking=no -o UserKnownHostsFile=$env:TEMP\runpod_known_hosts `
      "root@${sshHost}:/workspace/smoke-train.log" $logLocal 2>$null
    if (-not (Test-Path $logLocal)) { continue }
    $lines = Get-Content $logLocal -ErrorAction SilentlyContinue
    foreach ($line in $lines) {
      if ($line -match 'throughput/device/TPS[:\s]+([0-9.]+)') {
        $throughput += [double]$Matches[1]
      }
      if ($line -match 'step[=\s]+(\d+)') {
        $seenSteps[$Matches[1]] = $true
      }
      if ($line -match 'Training complete|max_duration reached|finished training') {
        break
      }
    }
    $stepCount = $seenSteps.Keys.Count
    $lastTps = if ($throughput.Count) { $throughput[-1] } else { 0 }
    Write-Host ("poll steps_seen=$stepCount last_TPS=$lastTps log_bytes=" + (Get-Item $logLocal).Length)
    if ($stepCount -ge 6 -or ($throughput.Count -ge 3 -and $stepCount -ge 3)) {
      Write-Host "Smoke target reached; stopping training..."
      Invoke-Ssh $sshHost $sshPort "pkill -f 'torch.distributed.run.*entrypoint.py' || true"
      break
    }
  }

  if (Test-Path $logLocal) {
    Write-Host "--- tail smoke-train.log ---"
    Get-Content $logLocal -Tail 40
    $tpsLines = Select-String -Path $logLocal -Pattern 'throughput|TPS|tokens/s' | Select-Object -Last 10
    if ($tpsLines) {
      Write-Host "--- throughput lines ---"
      $tpsLines | ForEach-Object { $_.Line }
    }
  }
}
finally {
  if ($podId) {
    Write-Host "Terminating pod $podId..."
    node $createPod --delete $podId
  }
  Remove-Item -Force $awsEnv, $wandbEnv -ErrorAction SilentlyContinue
}
