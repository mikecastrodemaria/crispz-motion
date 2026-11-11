# crispz-motion - update (Windows PowerShell): git pull --ff-only (git refuses to
# overwrite a file changed here), then reinstalls only the dependencies whose file
# changed; torch must stay a CUDA build. config.json is never touched: new keys are listed.
#   -ForceDeps   reinstall the dependencies anyway
# If scripts are blocked:  powershell -ExecutionPolicy Bypass -File update.ps1
param([switch]$ForceDeps, [switch]$NoPause)
Set-Location -LiteralPath $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    & (Join-Path $PSScriptRoot "install.ps1") -NoPause:$NoPause
    exit $LASTEXITCODE
}
$extra = @()
if ($ForceDeps) { $extra += "--force-deps" }
& $py cm_setup.py update @extra
$rc = $LASTEXITCODE
if ($rc -ne 0) { Write-Host "[update] FAILED, see the messages above" }
if (-not $NoPause) { Read-Host "Press Enter" | Out-Null }
exit $rc
