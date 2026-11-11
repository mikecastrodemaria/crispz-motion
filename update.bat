@echo off
REM crispz-motion - update (Windows): git pull --ff-only (git refuses to overwrite a file
REM changed here), then reinstalls only the dependencies whose file changed; torch must
REM stay a CUDA build. config.json is never touched: new keys are listed.
REM   --force-deps   reinstall the dependencies anyway
REM One block on purpose: the pull may change this very file while cmd reads it.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    call "%~dp0install.bat" %*
    exit /b
)
(
    .venv\Scripts\python.exe cm_setup.py update %*
    if errorlevel 1 echo [update] FAILED, see the messages above
    if /I not "%~1"=="--no-pause" pause
    exit /b
)
