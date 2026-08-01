# Launch Co-LMLM ModernBERT annotate smoke on a single RunPod A100.
# Uploads local model + 200-doc fixture over SSH (no S3).
param(
  [string]$GpuType = "NVIDIA A100-SXM4-80GB",
  [string]$PodName = "colmlm-annotate-smoke-a100",
  [string]$ModelDir = "C:\Users\natha\data\colmlm-1b-run\final",
  [string]$SmokeInput = "C:\Users\natha\data\colmlm-annotate-smoke",
  [int]$WaitSeconds = 900
)

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$repoRoot = "C:\alpha_ai\edullm"
if (-not (Test-Path (Join-Path $scriptDir "annotate_modernbert.py"))) {
  throw "annotate_modernbert.py not found under $scriptDir"
}
$sshKey = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$sshOpts = @(
  "-i", $sshKey,
  "-o", "StrictHostKeyChecking=no",
  "-o", "UserKnownHostsFile=NUL",
  "-o", "ConnectTimeout=10",
  "-o", "ServerAliveInterval=30"
)

if (-not (Test-Path $sshKey)) { throw "missing SSH key: $sshKey" }
if (-not (Test-Path (Join-Path $ModelDir "config.json"))) { throw "missing model: $ModelDir" }
if (-not (Test-Path $SmokeInput)) { throw "missing smoke input: $SmokeInput" }

Write-Host "Creating pod $PodName ($GpuType) ..." -ForegroundColor Cyan
$createOut = & node (Join-Path $scriptDir "create_annotate_smoke_pod.js") $GpuType $PodName
$createOut | ForEach-Object { Write-Host $_ }
$podId = ($createOut | Where-Object { $_ -match "^podId\s+(\S+)" } | ForEach-Object {
  if ($_ -match "^podId\s+(\S+)") { $Matches[1] }
}) | Select-Object -First 1
if (-not $podId) { throw "failed to create pod; no podId in create output" }
Write-Host "podId=$podId" -ForegroundColor Green

function Get-PodSsh {
  param([string]$Id)
  $mcp = Get-Content (Join-Path $env:USERPROFILE ".cursor\mcp.json") -Raw | ConvertFrom-Json
  $apiKey = $mcp.mcpServers.runpod.env.RUNPOD_API_KEY
  $headers = @{ Authorization = "Bearer $apiKey" }
  $pod = Invoke-RestMethod -Uri "https://rest.runpod.io/v1/pods/$Id" -Headers $headers
  $ssh = $null
  if ($pod.runtime -and $pod.runtime.ports) {
    $ssh = $pod.runtime.ports | Where-Object { $_.type -eq "tcp" -and $_.private -eq 22 } | Select-Object -First 1
  }
  [pscustomobject]@{
    Status = $pod.desiredStatus
    PublicIp = if ($ssh) { $ssh.ip } else { $null }
    PublicPort = if ($ssh) { $ssh.public } else { $null }
    Raw = $pod
  }
}

Write-Host "Waiting for SSH ..." -ForegroundColor Cyan
$deadline = (Get-Date).AddSeconds($WaitSeconds)
$sshHost = $null
$sshPort = $null
while ((Get-Date) -lt $deadline) {
  try {
    $info = Get-PodSsh -Id $podId
    Write-Host ("  status={0} ip={1} port={2}" -f $info.Status, $info.PublicIp, $info.PublicPort)
    if ($info.PublicIp -and $info.PublicPort) {
      $sshHost = $info.PublicIp
      $sshPort = $info.PublicPort
      # probe
      & ssh @sshOpts -p $sshPort "root@${sshHost}" "echo SSH_OK && nvidia-smi -L" 2>$null
      if ($LASTEXITCODE -eq 0) { break }
    }
  } catch {
    Write-Host "  poll error: $_"
  }
  $sshHost = $null
  Start-Sleep -Seconds 15
}
if (-not $sshHost) { throw "SSH not ready within ${WaitSeconds}s for pod $podId" }

$remote = "root@${sshHost}"
$remoteRoot = "/workspace/colmlm_annotate"
Write-Host "Uploading code + smoke input + model (~1.5 GiB) ..." -ForegroundColor Cyan

& ssh @sshOpts -p $sshPort $remote "mkdir -p $remoteRoot/model/final $remoteRoot/input $remoteRoot/output"
if ($LASTEXITCODE -ne 0) { throw "remote mkdir failed" }

# Code
& scp @sshOpts -P $sshPort `
  (Join-Path $scriptDir "annotate_modernbert.py") `
  (Join-Path $scriptDir "run_smoke.sh") `
  "${remote}:${remoteRoot}/"
if ($LASTEXITCODE -ne 0) { throw "scp code failed" }

# Smoke input (tiny)
& scp @sshOpts -P $sshPort -r `
  "${SmokeInput}\." `
  "${remote}:${remoteRoot}/input/"
if ($LASTEXITCODE -ne 0) { throw "scp smoke input failed" }

# Model final/
& scp @sshOpts -P $sshPort -r `
  "${ModelDir}\." `
  "${remote}:${remoteRoot}/model/final/"
if ($LASTEXITCODE -ne 0) { throw "scp model failed" }

& ssh @sshOpts -p $sshPort $remote "chmod +x $remoteRoot/run_smoke.sh && touch $remoteRoot/START_SMOKE"
if ($LASTEXITCODE -ne 0) { throw "failed to start smoke" }

Write-Host "Smoke started; waiting for SMOKE_DONE ..." -ForegroundColor Green
Write-Host "podId=$podId  ssh -i $sshKey -p $sshPort $remote"
$doneDeadline = (Get-Date).AddSeconds(1800)
while ((Get-Date) -lt $doneDeadline) {
  $log = & ssh @sshOpts -p $sshPort $remote "tail -n 40 $remoteRoot/smoke.log 2>/dev/null || true"
  $log | ForEach-Object { Write-Host $_ }
  if ($log -match "SMOKE_DONE|SMOKE_EXIT=") { break }
  Start-Sleep -Seconds 20
}
Write-Host "Done. Inspect $remoteRoot/output on the pod; terminate pod $podId when finished." -ForegroundColor Green
