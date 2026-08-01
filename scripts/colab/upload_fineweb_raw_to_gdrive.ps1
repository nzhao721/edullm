# Upload local FineWeb-Edu 1B raw text shards to gdrive-colab (same account/folder tree as tokenized).
param(
    [string]$SourceDir = "C:\Users\natha\data\fineweb-edu-1b-smollm2-raw",
    [string]$Remote = "gdrive-colab",
    [string]$DestPath = "edullm/fineweb-edu-1b-smollm2-raw",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path", "User")

if (-not (Test-Path $SourceDir)) {
    throw "Source not found: $SourceDir"
}

$dest = "${Remote}:${DestPath}"
Write-Host "Source:  $SourceDir"
Write-Host "Dest:    $dest"
Write-Host ""

$args = @(
    "copy", $SourceDir, $dest,
    "--progress",
    "--transfers", "4",
    "--checkers", "8",
    "--drive-chunk-size", "64M",
    "--stats", "30s",
    "--stats-one-line"
)

if ($DryRun) { $args += "--dry-run" }

& rclone @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Verifying remote listing..." -ForegroundColor Cyan
& rclone ls "${dest}" --max-depth 2
Write-Host ""
Write-Host "Done. In Colab (same Google account):" -ForegroundColor Green
Write-Host "  /content/drive/MyDrive/$DestPath"
