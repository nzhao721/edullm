# Mint FarmShare-style AWS session, stage local MixLaw code to S3, launch 8xA100 on RunPod.
# No GitHub: the pod downloads the local code tarball from S3.
param(
  [string]$Profile = "sbsandbox",
  [string]$GpuType = "NVIDIA A100-SXM4-80GB",
  [int]$GpuCount = 8,
  [int]$VolumeGb = 250,
  [int]$DeviceBatchSize = 16,
  [string]$CodeS3Uri = "s3://edullm-checkpoints/runpod/mixlaw-local-code.tgz"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$awsEnv = Join-Path $env:TEMP "aws-session-runpod.env"
$mintScript = Join-Path $repoRoot "scripts\farmshare\mint_aws_session_local.ps1"
$runScript = Join-Path $repoRoot "scripts\runpod\run_mixlaw_370m_mix01.sh"
$createPod = Join-Path $repoRoot "scripts\runpod\create_runpod.js"
$bundleDir = Join-Path $PSScriptRoot "_mixlaw_bundle"
$tarball = Join-Path $PSScriptRoot "mixlaw_local_code.tgz"

& $mintScript -Profile $Profile -OutputPath $awsEnv

# Source temporary session into this process for the code upload only.
Get-Content $awsEnv | ForEach-Object {
  if ($_ -match "^export ([A-Z0-9_]+)='(.*)'$") {
    Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
  }
}

if (-not (Test-Path $tarball)) {
  throw "missing local code tarball: $tarball (rebuild _mixlaw_bundle first)"
}

Write-Host "Uploading local MixLaw code → $CodeS3Uri"
aws s3 cp $tarball $CodeS3Uri --only-show-errors
Write-Host "code_staged $CodeS3Uri"

node $createPod $awsEnv $runScript $GpuType "mixlaw-370m-mix01-8xa100" $GpuCount $VolumeGb $DeviceBatchSize $CodeS3Uri
