# One capacity tick for curriculum linear10-learn (arm 2) on RunPod 8xA100.
# Rejects defective hosts before bootstrap/training. Exit 0 = training started.
param(
  [string]$Profile = "sbsandbox",
  [int]$ArmIndex = 2,
  [int]$VolumeGb = 250,
  [string]$OlmoCoreRoot = "C:\alpha_ai\OLMo-core-curriculum-370m",
  [string]$OlmoCoreCommit = "740d2bae39d2bbea3299c55e19a603dab043a839",
  [string[]]$BlockedSshHosts = @("154.54.102.51", "154.54.102.44")
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$createPod = Join-Path $PSScriptRoot "smollm2_colmlm\create_idle_pod.js"
$mintScript = Join-Path $repoRoot "scripts\farmshare\mint_aws_session_local.ps1"
$wandbKeyFile = Join-Path $env:USERPROFILE ".wandb_api_key"
$sshKey = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$sshPub = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519.pub"
$awsEnv = Join-Path $env:TEMP "aws-session-curriculum-learn.env"
$wandbEnv = Join-Path $env:TEMP "wandb-session-curriculum-learn.env"
$logLocal = Join-Path $env:TEMP "curriculum-learn-train.log"
$podName = "curriculum-linear10-learn-370m-8xa100"
$repoDir = "/workspace/OLMo-core"
$trainLog = "/workspace/curriculum-learn-train.log"

$createAttempts = @(
  @{ GpuType = "NVIDIA A100-SXM4-80GB"; CloudType = "COMMUNITY" },
  @{ GpuType = "NVIDIA A100-SXM4-80GB"; CloudType = "SECURE" },
  @{ GpuType = "NVIDIA A100 80GB PCIe"; CloudType = "COMMUNITY" },
  @{ GpuType = "NVIDIA A100 80GB PCIe"; CloudType = "SECURE" }
)

if (-not (Test-Path $OlmoCoreRoot)) { throw "OLMo-core worktree missing: $OlmoCoreRoot" }
if (-not (Test-Path (Join-Path $OlmoCoreRoot ".edullm\runpod\launch.sh"))) {
  throw "missing runpod adapter under $OlmoCoreRoot\.edullm\runpod"
}

function Test-BlockedHost {
  param([string]$SshHost)
  return $BlockedSshHosts -contains $SshHost
}

function Invoke-Ssh {
  param([string]$HostName, [int]$Port, [string]$RemoteCommand)
  & ssh -i $sshKey -p $Port -o StrictHostKeyChecking=no -o UserKnownHostsFile=$env:TEMP\runpod_known_hosts -o BatchMode=yes -o ConnectTimeout=20 "root@${HostName}" $RemoteCommand
  if ($LASTEXITCODE -ne 0) { throw "ssh failed: $RemoteCommand" }
}

function Invoke-Scp {
  param([string]$LocalPath, [string]$HostName, [int]$Port, [string]$RemotePath)
  & scp -i $sshKey -P $Port -o StrictHostKeyChecking=no -o UserKnownHostsFile=$env:TEMP\runpod_known_hosts $LocalPath "root@${HostName}:$RemotePath"
  if ($LASTEXITCODE -ne 0) { throw "scp failed: $LocalPath -> $RemotePath" }
}

function Remove-LearnPod {
  param([string]$PodId, [string]$Reason)
  Write-Host "Deleting pod $PodId ($Reason)..."
  node $createPod --delete $PodId | Out-Null
}

function Get-TrainingStepCount {
  param([string]$HostName, [int]$Port)
  try {
    $out = & ssh -i $sshKey -p $Port -o StrictHostKeyChecking=no -o UserKnownHostsFile=$env:TEMP\runpod_known_hosts -o BatchMode=yes -o ConnectTimeout=15 "root@${HostName}" "test -f '$trainLog' && grep -Eo 'step[=[:space:]]+[0-9]+' '$trainLog' | tail -1 || true"
    if ($out -match 'step[=[:space:]]+(\d+)') { return [int]$Matches[1] }
  } catch {}
  return 0
}

$listJson = node $createPod --list $podName
$parsed = if ($listJson) { $listJson | ConvertFrom-Json } else { $null }
$existing = if ($null -eq $parsed) { @() } elseif ($parsed -is [System.Array]) { $parsed } else { @($parsed) }
foreach ($pod in $existing) {
  $inspectJson = node $createPod --inspect $pod.id
  $info = $inspectJson | ConvertFrom-Json
  $sshHost = $info.sshHost
  $sshPort = [int]$info.sshPort
  if (-not $sshHost -or -not $sshPort) {
    Write-Host "Existing pod $($pod.id) has no SSH yet; skipping this tick."
    exit 3
  }
  if (Test-BlockedHost $sshHost) {
    Remove-LearnPod $pod.id "blocked or unstable host $sshHost"
    continue
  }
  $step = Get-TrainingStepCount $sshHost $sshPort
  if ($step -ge 1) {
    Write-Host "Training already running on $($sshHost):$sshPort (step=$step, pod=$($pod.id))."
    exit 0
  }
  Write-Host "Found idle learn pod $($pod.id) on good host $($sshHost):$sshPort; continuing bootstrap."
  $script:podId = $pod.id
  $script:sshHost = $sshHost
  $script:sshPort = $sshPort
  break
}

$createdHere = $false
if (-not $script:podId) {
  $created = $false
  foreach ($attempt in $createAttempts) {
    Write-Host "Trying 8x $($attempt.GpuType) ($($attempt.CloudType))..."
    try {
      $createJson = node $createPod $attempt.GpuType $podName 8 $VolumeGb $sshPub $attempt.CloudType 2>&1
      if ($LASTEXITCODE -ne 0) { continue }
      $podInfo = ($createJson | Select-Object -Last 1) | ConvertFrom-Json
      $script:podId = $podInfo.podId
      $script:sshHost = $podInfo.sshHost
      $script:sshPort = [int]$podInfo.sshPort
      $created = $true
      $createdHere = $true
      break
    } catch {
      Write-Host "Create failed: $_"
    }
  }
  if (-not $created) {
    Write-Host "No 8xA100 capacity on any tier."
    exit 3
  }
}

if (Test-BlockedHost $sshHost) {
  Remove-LearnPod $podId "blocked host $sshHost before bootstrap"
  exit 2
}

Write-Host "Using pod $podId at ${sshHost}:$sshPort"

try {
  Write-Host "Minting temporary AWS session (laptop)..."
  & $mintScript -Profile $Profile -OutputPath $awsEnv

  Write-Host "Preparing clean checkout at $repoDir..."
  Invoke-Ssh $sshHost $sshPort "rm -rf '$repoDir' /workspace/edullm-inputs/curriculum /workspace/edullm-runs/curriculum"

  Write-Host "Running bootstrap.sh on pod (commit $OlmoCoreCommit)..."
  $bootstrap = Join-Path $OlmoCoreRoot ".edullm\runpod\bootstrap.sh"
  Invoke-Scp $bootstrap $sshHost $sshPort "/tmp/bootstrap.sh"
  Invoke-Ssh $sshHost $sshPort "chmod +x /tmp/bootstrap.sh && REPO_DIR='$repoDir' OLMO_CORE_COMMIT_SHA='$OlmoCoreCommit' bash /tmp/bootstrap.sh"

  Write-Host "Copying local runpod adapter..."
  Invoke-Ssh $sshHost $sshPort "mkdir -p '$repoDir/.edullm/runpod'"
  & scp -i $sshKey -P $sshPort -o StrictHostKeyChecking=no -o UserKnownHostsFile=$env:TEMP\runpod_known_hosts -r `
    (Join-Path $OlmoCoreRoot ".edullm\runpod\*") "root@${sshHost}:$repoDir/.edullm/runpod/"
  if ($LASTEXITCODE -ne 0) { throw "scp runpod adapter failed" }

  Write-Host "Staging sealed inputs (arm $ArmIndex)..."
  Invoke-Scp $awsEnv $sshHost $sshPort "/workspace/aws-session.env"
  Invoke-Ssh $sshHost $sshPort "chmod 600 /workspace/aws-session.env"
  Invoke-Ssh $sshHost $sshPort @"
cd '$repoDir' && PYTHONPATH='$repoDir/src:$repoDir/.edullm' python3 .edullm/runpod/stage_inputs.py --credentials-file /workspace/aws-session.env --arm-index $ArmIndex --curriculum-version v1
"@

  if (-not (Test-Path $wandbKeyFile)) { throw "missing W&B key file: $wandbKeyFile" }
  $wandbKey = (Get-Content $wandbKeyFile -Raw).Trim()
  @"
# Generated for RunPod curriculum learn. Do not commit.
export WANDB_API_KEY='$wandbKey'
export WANDB_ENTITY='eduLLM'
export WANDB_PROJECT='curriculum'
export WANDB_START_METHOD=thread
"@ | Set-Content -NoNewline -Encoding ascii $wandbEnv
  Invoke-Scp $wandbEnv $sshHost $sshPort "/workspace/wandb-session.env"
  Invoke-Ssh $sshHost $sshPort "chmod 600 /workspace/wandb-session.env"

  Write-Host "Launching full training (arm $ArmIndex, project curriculum)..."
  $launchCmd = "cd '$repoDir' && REPO_DIR='$repoDir' ARM_INDEX=$ArmIndex EDULLM_WANDB_PROJECT=curriculum nohup bash .edullm/runpod/launch.sh > '$trainLog' 2>&1 < /dev/null & echo started"
  $startOut = Invoke-Ssh $sshHost $sshPort $launchCmd
  Write-Host $startOut

  $deadline = (Get-Date).AddMinutes(60)
  while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 30
    & scp -i $sshKey -P $sshPort -o StrictHostKeyChecking=no -o UserKnownHostsFile=$env:TEMP\runpod_known_hosts `
      "root@${sshHost}:$trainLog" $logLocal 2>$null
    if (-not (Test-Path $logLocal)) { continue }
    $step = 0
    foreach ($line in (Get-Content $logLocal -ErrorAction SilentlyContinue)) {
      if ($line -match 'step[=\s]+(\d+)') { $step = [int]$Matches[1] }
    }
    Write-Host "poll step=$step log_bytes=$((Get-Item $logLocal).Length)"
    if ($step -ge 1) {
      Write-Host "First training step observed; tick complete."
      Get-Content $logLocal -Tail 20
      exit 0
    }
    if (Select-String -Path $logLocal -Pattern 'Traceback|RuntimeError|CUDA error' -Quiet) {
      throw "training failed early; see $logLocal"
    }
  }
  throw "timed out waiting for first training step"
}
catch {
  if ($createdHere -and $podId) {
    Remove-LearnPod $podId "bootstrap/launch failure: $_"
  }
  throw
}
finally {
  Remove-Item -Force $awsEnv, $wandbEnv -ErrorAction SilentlyContinue
}
