# Mint temporary AWS credentials on this Windows device and write aws-session.env
# for an approved FarmShare or RunPod SSH bootstrap. Never prints secret material.
param(
  [string]$Profile = "sbsandbox",
  [string]$Region = "us-east-1",
  [Parameter(Mandatory = $true)]
  [string]$OutputPath
)

$ErrorActionPreference = "Stop"

$raw = & aws configure export-credentials --profile $Profile --format process
if ($LASTEXITCODE -ne 0) {
  throw "aws configure export-credentials failed (exit $LASTEXITCODE). Run: sb-aws-creds login"
}

$creds = $raw | ConvertFrom-Json
if (-not $creds.AccessKeyId -or -not $creds.SecretAccessKey -or -not $creds.SessionToken) {
  throw "export-credentials response missing required fields"
}

$dir = Split-Path -Parent $OutputPath
if ($dir -and -not (Test-Path $dir)) {
  New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

function Quote-Bash([string]$value) {
  return "'" + ($value -replace "'", "'\''") + "'"
}

$lines = @(
  "# Generated on engineer laptop for approved remote bootstrap; do not commit or log.",
  "unset AWS_PROFILE",
  ("export AWS_ACCESS_KEY_ID=" + (Quote-Bash $creds.AccessKeyId)),
  ("export AWS_SECRET_ACCESS_KEY=" + (Quote-Bash $creds.SecretAccessKey)),
  ("export AWS_SESSION_TOKEN=" + (Quote-Bash $creds.SessionToken)),
  ("export AWS_REGION=" + (Quote-Bash $Region)),
  ("export AWS_DEFAULT_REGION=" + (Quote-Bash $Region))
)
if ($creds.Expiration) {
  $lines += ("# Expiration=" + $creds.Expiration)
}

$utf8NoBom = New-Object System.Text.UTF8Encoding $false
$content = ($lines -join "`n") + "`n"
[System.IO.File]::WriteAllText($OutputPath, $content, $utf8NoBom)

$suffix = $creds.AccessKeyId.Substring($creds.AccessKeyId.Length - 4)
Write-Host "aws_session_written key=...$suffix expiry=$($creds.Expiration) path=$OutputPath"
