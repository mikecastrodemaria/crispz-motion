# crispz-motion - install (Windows PowerShell). The work is in cm_setup.py:
#   .venv the app (Pillow), film\.venv FILM + ESRGAN (torch cu128, spandrel),
#   film\models\film_net_fp32.pt (138 MB, Apache 2.0), config.json when missing.
# If scripts are blocked:  powershell -ExecutionPolicy Bypass -File install.ps1
param([switch]$NoPause)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$py = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($v in "3.11", "3.12", "3.13", "3.10") {
        & py "-$v" -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break }
    }
}
if (-not $py) { $py = @("python") }
Write-Host "[install] Python: $($py -join ' ')"
$exe = $py[0]
$pre = @($py | Select-Object -Skip 1)
& $exe @pre cm_setup.py install
$rc = $LASTEXITCODE
if ($rc -ne 0) { Write-Host "[install] FAILED, see the messages above" }
if (-not $NoPause) { Read-Host "Press Enter" | Out-Null }
exit $rc
