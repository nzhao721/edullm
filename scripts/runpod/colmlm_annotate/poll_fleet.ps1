# Poll Co-LMLM annotate fleet: RunPod GPU util + S3 completed outputs.
$ErrorActionPreference = "Continue"
$fleetPath = "C:\alpha_ai\edullm\scripts\runpod\colmlm_annotate\fleet_2026-08-01.json"
$mcp = Get-Content "$env:USERPROFILE\.cursor\mcp.json" -Raw | ConvertFrom-Json
$apiKey = $mcp.mcpServers.runpod.env.RUNPOD_API_KEY
$headers = @{ Authorization = "Bearer $apiKey" }
$fleet = Get-Content $fleetPath -Raw | ConvertFrom-Json

$running = 0; $gpuActive = 0; $err = 0
$rows = @()
foreach ($w in $fleet.workers) {
  try {
    $p = Invoke-RestMethod -Uri "https://rest.runpod.io/v1/pods/$($w.podId)" -Headers $headers
    $status = $p.status
    if (-not $status) { $status = $p.desiredStatus }
    $gpu = 0; $mem = 0; $up = 0
    if ($p.runtime -and $p.runtime.gpus -and $p.runtime.gpus.Count -gt 0) {
      $gpu = [int]$p.runtime.gpus[0].util
      $mem = [int]$p.runtime.gpus[0].memoryUtil
    }
    if ($p.runtime -and $p.runtime.uptime) { $up = [int]$p.runtime.uptime }
    if ($status -eq "RUNNING") { $running++ }
    if ($gpu -gt 5) { $gpuActive++ }
    $rows += [pscustomobject]@{ w = $w.workerIndex; status = $status; gpu = $gpu; mem = $mem; upMin = [math]::Round($up / 60, 1) }
  } catch {
    $err++
    $rows += [pscustomobject]@{ w = $w.workerIndex; status = "ERR"; gpu = 0; mem = 0; upMin = 0 }
  }
}

# S3 completed workers (via sb-aws would be ideal; use local aws profile for listing only if available)
$done = @()
try {
  $listing = & aws s3 ls "s3://edullm-checkpoints/runpod/colmlm-annotate/output/" --profile sbsandbox 2>$null
  if ($listing) {
    $done = @($listing | ForEach-Object {
      if ($_ -match "worker-(\d+)/") { [int]$Matches[1] }
    } | Sort-Object -Unique)
  }
} catch {}

$ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
Write-Host "FLEET_POLL $ts running=$running/19 gpuActive=$gpuActive s3Done=$($done.Count)/19 err=$err"
if ($done.Count -gt 0) { Write-Host ("s3DoneWorkers=" + ($done -join ",")) }
$rows | Sort-Object w | Format-Table -AutoSize | Out-String | Write-Host
$idle = @($rows | Where-Object { $_.status -eq "RUNNING" -and $_.gpu -le 5 } | ForEach-Object { $_.w })
if ($idle.Count) { Write-Host ("lowGpuWorkers=" + ($idle -join ",")) }
