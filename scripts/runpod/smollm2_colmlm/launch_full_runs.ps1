# Launch one or more full SmolLM2 Co-LMLM runs with a single temporary AWS session.
param(
  [string]$Profile = "sbsandbox",
  [string[]]$GpuTypes = @("NVIDIA L40S", "NVIDIA A100-SXM4-80GB"),
  [int]$GpuCount = 4,
  [int]$VolumeGb = 80,
  [int]$PerDeviceBatchSize = 16,
  [string]$WandbProject = "edullm-smollm2",
  [ValidateSet("COMMUNITY", "SECURE")]
  [string]$CloudType = "SECURE",
  [string]$SshKeyPath = (Join-Path $HOME ".ssh\edullm_runpod")
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
$mintScript = Join-Path $repoRoot "scripts\farmshare\mint_aws_session_local.ps1"
$createPod = Join-Path $PSScriptRoot "create_idle_pod.js"
$tempRoot = Join-Path $env:TEMP ("smollm2-colmlm-" + [Guid]::NewGuid().ToString("N"))
$awsEnv = Join-Path $tempRoot "aws-session.env"
$wandbEnv = Join-Path $tempRoot "wandb-session.env"
$bundle = Join-Path $tempRoot "smollm2-colmlm.tgz"
$knownHosts = Join-Path $tempRoot "known_hosts"
$launched = @()
$prepared = @()
$success = $false

function Get-WandbKey {
  if ($env:WANDB_API_KEY) {
    return $env:WANDB_API_KEY.Trim()
  }
  $candidate = Join-Path $HOME ".wandb_api_key"
  if (Test-Path $candidate) {
    return (Get-Content -Raw $candidate).Trim()
  }
  throw "WANDB_API_KEY missing (set it or create $candidate)"
}

function Quote-Bash([string]$Value) {
  return "'" + ($Value -replace "'", "'\''") + "'"
}

try {
  New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
  if (-not (Test-Path $SshKeyPath) -or -not (Test-Path "$SshKeyPath.pub")) {
    New-Item -ItemType Directory -Path (Split-Path $SshKeyPath -Parent) -Force | Out-Null
    & ssh-keygen -q -t ed25519 -N '""' -f $SshKeyPath
    if ($LASTEXITCODE -ne 0) {
      throw "failed to create dedicated RunPod SSH key at $SshKeyPath"
    }
  }
  $wandbKey = Get-WandbKey
  $wandbBody = @(
    "# Temporary W&B session for RunPod; do not commit or log."
    ("export WANDB_API_KEY=" + (Quote-Bash $wandbKey))
    "export WANDB_START_METHOD=thread"
    ("export WANDB_PROJECT=" + (Quote-Bash $WandbProject))
    "export WANDB_GROUP='colmlm-fact-masked'"
    "export WANDB_MODE=online"
  ) -join "`n"
  [IO.File]::WriteAllText(
    $wandbEnv,
    $wandbBody + "`n",
    (New-Object Text.UTF8Encoding $false)
  )

  & tar -czf $bundle -C $PSScriptRoot .
  if ($LASTEXITCODE -ne 0) {
    throw "failed to create code bundle"
  }

  # Provision, upload code, and install packages before minting AWS credentials.
  foreach ($gpuType in $GpuTypes) {
    $slug = ($gpuType.ToLowerInvariant() -replace "[^a-z0-9]+", "-").Trim("-")
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")
    $podName = "smollm2-colmlm-$GpuCount`x$slug"
    $runName = "$podName-$stamp"
    Write-Host "Creating full-run pod: $podName"
    $podJson = & node $createPod $gpuType $podName $GpuCount $VolumeGb "$SshKeyPath.pub" $CloudType
    if ($LASTEXITCODE -ne 0) {
      throw "RunPod creation failed for $gpuType"
    }
    $pod = $podJson | ConvertFrom-Json
    $target = "root@$($pod.sshHost)"
    $sshArgs = @(
      "-p", "$($pod.sshPort)",
      "-i", $SshKeyPath,
      "-o", "BatchMode=yes",
      "-o", "StrictHostKeyChecking=accept-new",
      "-o", "UserKnownHostsFile=$knownHosts",
      $target
    )
    $scpArgs = @(
      "-P", "$($pod.sshPort)",
      "-i", $SshKeyPath,
      "-o", "BatchMode=yes",
      "-o", "StrictHostKeyChecking=accept-new",
      "-o", "UserKnownHostsFile=$knownHosts"
    )
    $prepared += [PSCustomObject]@{
      podId = $pod.podId
      name = $podName
      runName = $runName
      gpuType = $gpuType
      gpuCount = $GpuCount
      costPerHr = $pod.costPerHr
      sshHost = $pod.sshHost
      sshPort = $pod.sshPort
      target = $target
      sshArgs = $sshArgs
      scpArgs = $scpArgs
    }

    # Retry while the image's SSH service finishes starting.
    $ready = $false
    for ($attempt = 1; $attempt -le 30; $attempt++) {
      & ssh @sshArgs "mkdir -p /workspace/bootstrap /workspace/edullm-smollm2-colmlm"
      if ($LASTEXITCODE -eq 0) {
        $ready = $true
        break
      }
      Start-Sleep -Seconds 5
    }
    if (-not $ready) {
      throw "SSH did not become ready for pod $($pod.podId)"
    }

    & scp @scpArgs $bundle "${target}:/workspace/bootstrap/code.tgz"
    if ($LASTEXITCODE -ne 0) { throw "code upload failed for $($pod.podId)" }
    $prepareRemote = @(
      "set -Eeuo pipefail"
      "tar -xzf /workspace/bootstrap/code.tgz -C /workspace/edullm-smollm2-colmlm"
      "sed -i 's/\r$//' /workspace/edullm-smollm2-colmlm/run_full_training.sh"
      "chmod +x /workspace/edullm-smollm2-colmlm/run_full_training.sh"
      "rm -rf /opt/edullm-venv"
      "python3 -m venv --system-site-packages /opt/edullm-venv"
      "/opt/edullm-venv/bin/python -m pip install -q --no-cache-dir -U pip wheel"
      "for attempt in 1 2 3; do /opt/edullm-venv/bin/python -m pip install -q --no-cache-dir 'transformers>=4.48,<5' 'wandb>=0.17' numpy zstandard awscli && break; [[ `$attempt -eq 3 ]] && exit 1; sleep 10; done"
      "/opt/edullm-venv/bin/python -c 'import numpy, torch, transformers, wandb, zstandard'"
      "/opt/edullm-venv/bin/aws --version >/dev/null"
      "echo prepared"
    ) -join "`n"
    & ssh @sshArgs $prepareRemote
    if ($LASTEXITCODE -ne 0) {
      throw "remote dependency preparation failed on $($pod.podId)"
    }

  }

  # Mint once, immediately before the bounded startup download on every pod.
  & $mintScript -Profile $Profile -OutputPath $awsEnv
  if ($LASTEXITCODE -ne 0) {
    throw "AWS session mint failed"
  }

  foreach ($pod in $prepared) {
    $podScpArgs = @($pod.scpArgs)
    $podSshArgs = @($pod.sshArgs)
    & scp @podScpArgs $awsEnv "$($pod.target):/workspace/bootstrap/aws-session.env"
    if ($LASTEXITCODE -ne 0) { throw "AWS session upload failed for $($pod.podId)" }
    & scp @podScpArgs $wandbEnv "$($pod.target):/workspace/bootstrap/wandb-session.env"
    if ($LASTEXITCODE -ne 0) { throw "W&B session upload failed for $($pod.podId)" }

    $startRemote = @(
      "set -Eeuo pipefail"
      "chmod 600 /workspace/bootstrap/aws-session.env /workspace/bootstrap/wandb-session.env"
      ("nohup env RUN_NAME=" + (Quote-Bash $pod.runName) +
        " NPROC=" + $GpuCount +
        " PER_DEVICE_BATCH_SIZE=" + $PerDeviceBatchSize +
        " WANDB_PROJECT=" + (Quote-Bash $WandbProject) +
        " bash /workspace/edullm-smollm2-colmlm/run_full_training.sh" +
        " > /workspace/bootstrap/full-run.log 2>&1 < /dev/null &")
      "echo `$! > /workspace/bootstrap/full-run.pid"
      "echo started pid=`$(cat /workspace/bootstrap/full-run.pid)"
    ) -join "`n"
    & ssh @podSshArgs $startRemote
    if ($LASTEXITCODE -ne 0) {
      throw "remote full run failed to start on $($pod.podId)"
    }
    Write-Host "Started $($pod.runName) on pod $($pod.podId)"
  }
  Remove-Item -Force $awsEnv
  if (Test-Path $awsEnv) {
    throw "local AWS session file was not deleted after transfer"
  }

  # Do not return success until each pod has verified its S3 inputs and erased
  # the remote credential file.
  foreach ($pod in $prepared) {
    $podSshArgs = @($pod.sshArgs)
    $staged = $false
    $probeRemote = (@'
if [[ ! -f /workspace/bootstrap/aws-session.env ]] &&
   grep -q '\[stage\] AWS credentials deleted' /workspace/bootstrap/full-run.log &&
   grep -q '\[prepare\]' /workspace/bootstrap/full-run.log &&
   [[ -f /workspace/bootstrap/full-run.pid ]] &&
   kill -0 "$(cat /workspace/bootstrap/full-run.pid)" 2>/dev/null; then
  echo STAGED
elif [[ -f /workspace/bootstrap/full-run.pid ]] &&
     kill -0 "$(cat /workspace/bootstrap/full-run.pid)" 2>/dev/null; then
  echo WAIT
else
  rm -f /workspace/bootstrap/aws-session.env
  echo FAILED
fi
'@) -replace "`r", ""
    for ($attempt = 1; $attempt -le 180; $attempt++) {
      $state = & ssh @podSshArgs $probeRemote
      if ($LASTEXITCODE -ne 0) {
        Start-Sleep -Seconds 10
        continue
      }
      if ($state -match "STAGED") {
        $staged = $true
        break
      }
      if ($state -match "FAILED") {
        throw "full run exited before secure staging completed on $($pod.podId)"
      }
      Start-Sleep -Seconds 10
    }
    if (-not $staged) {
      throw "timed out waiting for secure staging on $($pod.podId)"
    }
    $launched += [PSCustomObject]@{
      podId = $pod.podId
      name = $pod.name
      runName = $pod.runName
      gpuType = $pod.gpuType
      gpuCount = $GpuCount
      costPerHr = $pod.costPerHr
      sshHost = $pod.sshHost
      sshPort = $pod.sshPort
    }
    Write-Host "Secure staging complete on pod $($pod.podId)"
  }

  $success = $true
  $launched | ConvertTo-Json -Depth 4
}
finally {
  if (-not $success) {
    foreach ($pod in $prepared) {
      $podSshArgs = @($pod.sshArgs)
      & ssh @podSshArgs "rm -f /workspace/bootstrap/aws-session.env" 2>$null
      & node $createPod --delete $pod.podId 2>$null
    }
  }
  # The same short-lived AWS session is copied once to each requested pod, then
  # removed from this device. Each pod removes its copy after its bounded sync.
  Remove-Item -Recurse -Force $tempRoot -ErrorAction SilentlyContinue
}
