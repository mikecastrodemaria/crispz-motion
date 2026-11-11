@echo off
REM crispz-motion - install (Windows). The work is in cm_setup.py:
REM   .venv       the app (Pillow)
REM   film\.venv  FILM + ESRGAN (torch cu128, spandrel)
REM   film\models\film_net_fp32.pt (138 MB, Apache 2.0)
REM   config.json from config-sample.json when missing
REM Python 3.10 to 3.13 (3.11 preferred).
setlocal
cd /d "%~dp0"
set "PY="
where py >nul 2>&1
if not errorlevel 1 (
    for %%V in (3.11 3.12 3.13 3.10) do (
        if not defined PY ( py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V" )
    )
)
if not defined PY set "PY=python"
echo [install] Python: %PY%
%PY% cm_setup.py install
if errorlevel 1 (
    echo [install] FAILED, see the messages above
    if /I not "%~1"=="--no-pause" pause
    exit /b 1
)
if /I not "%~1"=="--no-pause" pause
endlocal
