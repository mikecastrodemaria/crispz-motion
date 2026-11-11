@echo off
REM crispz-motion - start (Windows).
REM   1. first run: install
REM   2. GitHub has new commits: asks "Update first?" (10 s, no answer = no)
REM   3. no argument (a double-click): the web page (cm_server.py, Ctrl+C to stop);
REM      arguments: the command line (cm_cli.py), e.g. start.bat status
REM After an error the window waits for a key: otherwise it closes before anything can be read.
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
set "APP=cm_cli.py"
if "%~1"=="" set "APP=cm_server.py"
set "NEW=0"
for /f %%n in ('%PY% cm_setup.py check-update 2^>nul') do set "NEW=%%n"
if not "%NEW%"=="0" (
    echo [start] %NEW% new commit^(s^) on GitHub.
    choice /C YN /N /T 10 /D N /M "[start] Update first? [Y/N] (10 s, default N) "
    if not errorlevel 2 call "%~dp0update.bat" --no-pause
    %PY% %APP% %*
    if errorlevel 1 (
        echo [start] stopped with an error, see the messages above
        pause
    )
    exit /b
)
%PY% %APP% %*
if errorlevel 1 (
    echo [start] stopped with an error, see the messages above
    pause
    exit /b 1
)
endlocal
