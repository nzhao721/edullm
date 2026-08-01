# Upload local FineWeb SmolLM2 tokenized corpus to gdrive-colab (separate Google account).
param(
    [string]$SourceDir = "C:\Users\natha\data\fineweb-edu-1b-smollm2-tokenized",
    [string]$Remote = "gdrive-colab",
    [string]$DestPath = "edullm/fineweb-edu-1b-smollm2-tokenized",
    [switch]$TokensOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path", "User")

if (-not (Test-Path $SourceDir)) {
    throw "Source not found: $SourceDir"
}

$includes = @(
    "--include", "meta.json",
    "--include", "subsets.json",
    "--include", "SUBSETS.md",
    "--include", "val/**"
)
if ($TokensOnly) {
    $includes += @("--include", "train_tokens.bin")
} else {
    $includes += @(
        "--include", "train_tokens.bin",
        "--include", "train_doc_ids.bin",
        "--include", "train_positions.bin"
    )
}

$dest = "${Remote}:${DestPath}"
Write-Host "Source:  $SourceDir"
Write-Host "Dest:    $dest"
Write-Host "Mode:    $(if ($TokensOnly) { 'tokens-only (~3.8 GB)' } else { 'full train bins (~11 GB)' })"
Write-Host ""

$args = @(
    "copy", $SourceDir, $dest,
    "--progress",
    "--transfers", "4",
    "--checkers", "8",
    "--drive-chunk-size", "64M",
    "--stats", "30s",
    "--stats-one-line"
) + $includes + @("--exclude", "*")

if ($DryRun) { $args += "--dry-run" }

& rclone @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Verifying remote listing..." -ForegroundColor Cyan
& rclone ls "${dest}" --max-depth 2
Write-Host ""
Write-Host "Done. In Colab (same Google account): mount Drive and use" -ForegroundColor Green
Write-Host "  /content/drive/MyDrive/$DestPath"
