@echo off
REM crispz-motion - start (Windows).
REM   1. first run: install
REM   2. GitHub has new commits: asks "Update first?" (10 s, no answer = no)
REM   3. runs the app: the arguments go to cm_cli.py (none = status)
REM Started with no argument (a double-click), the window waits for a key at the end,
REM and it always does after an error: otherwise it closes before anything can be read.
REM Everything after the update check sits in ONE block: cmd reads a .bat while it runs,
REM and a git pull that changes this file would otherwise make it read the wrong line.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    call "%~dp0install.bat" --no-pause
    if errorlevel 1 (
        echo [start] the install failed, see the messages above
        pause
        exit /b 1
    )
)
set "PY=.venv\Scripts\python.exe"
set "NEW=0"
for /f %%n in ('%PY% cm_setup.py check-update 2^>nul') do set "NEW=%%n"
if not "%NEW%"=="0" (
    echo [start] %NEW% new commit^(s^) on GitHub.
    choice /C YN /N /T 10 /D N /M "[start] Update first? [Y/N] (10 s, default N) "
    if not errorlevel 2 call "%~dp0update.bat" --no-pause
    %PY% cm_cli.py %*
    if errorlevel 1 (
        echo [start] stopped with an error, see the messages above
        pause
    ) else if "%~1"=="" pause
    exit /b
)
%PY% cm_cli.py %*
if errorlevel 1 (
    echo [start] stopped with an error, see the messages above
    pause
    exit /b 1
)
if "%~1"=="" pause
endlocal
