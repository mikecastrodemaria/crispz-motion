# crispz-motion - start (Windows PowerShell).
#   1. first run: install
#   2. GitHub has new commits: asks "Update first?" (10 s, no answer = no)
#   3. no argument: the web page (cm_server.py, Ctrl+C to stop); arguments: cm_cli.py
# If scripts are blocked:  powershell -ExecutionPolicy Bypass -File start.ps1
Set-Location -LiteralPath $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    & (Join-Path $PSScriptRoot "install.ps1") -NoPause
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

$new = 0
$n = & $py cm_setup.py check-update 2>$null
if ("$n".Trim() -match '^\d+$') { $new = [int]"$n".Trim() }
if ($new -gt 0) {
    Write-Host "[start] $new new commit(s) on GitHub."
    Write-Host -NoNewline "[start] Update first? [y/N] (10 s, default N) "
    $answer = "n"
    $deadline = (Get-Date).AddSeconds(10)
    try {
        while ((Get-Date) -lt $deadline) {
            if ([Console]::KeyAvailable) { $answer = [string][Console]::ReadKey($true).KeyChar; break }
            Start-Sleep -Milliseconds 100
        }
    } catch { $answer = "n" }      # no console (input redirected): never block
    Write-Host $answer
    if ($answer -match '^[yYoO]$') {
        & (Join-Path $PSScriptRoot "update.ps1") -NoPause
    }
}
if ($args.Count -eq 0) { & $py cm_server.py } else { & $py cm_cli.py @args }
$rc = $LASTEXITCODE
# an error: keep the window open
if ($rc -ne 0) {
    Write-Host "[start] stopped with an error, see the messages above"
    Read-Host "Press Enter" | Out-Null
}
exit $rc
