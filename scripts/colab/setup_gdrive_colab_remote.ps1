# Configure rclone remote "gdrive-colab" for a Google account separate from Drive for Desktop.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Invoke-Rclone {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & rclone @Args
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($code -ne 0) { exit $code }
}

$rclone = Get-Command rclone -ErrorAction SilentlyContinue
if (-not $rclone) {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
    $rclone = Get-Command rclone -ErrorAction Stop
}

$remote = "gdrive-colab"
Write-Host "Creating rclone remote '$remote' (Google Drive)..." -ForegroundColor Cyan
Write-Host "A browser window will open. Sign in with the Google account for Colab uploads" `
     -ForegroundColor Yellow
Write-Host "(NOT the account used by Google Drive for Desktop)." -ForegroundColor Yellow
Write-Host ""

@('y', 'y', 'n') | rclone config reconnect "${remote}:" 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "OK. Testing Drive access..." -ForegroundColor Green
Invoke-Rclone lsd "${remote}:"

Write-Host ""
Write-Host "Remote '$remote' is ready. Run upload_fineweb_to_gdrive.ps1 next." -ForegroundColor Green
