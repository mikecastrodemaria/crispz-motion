@echo off
REM crispz-motion - FILM venv (film\.venv: torch cu128) + the FILM "Style" weights.
REM The app itself stays stdlib + Pillow; torch lives in this venv only.
setlocal
cd /d "%~dp0"

set FILM_TAG=v1.0.2
set FILM_FILE=film_net_fp32.pt
set FILM_URL=https://github.com/dajes/frame-interpolation-pytorch/releases/download/%FILM_TAG%/%FILM_FILE%
set FILM_SIZE=137922129

where py >nul 2>&1
if errorlevel 1 ( set PYCMD=python ) else (
    py -3.11 -c "import sys" >nul 2>&1
    if errorlevel 1 ( set PYCMD=py -3 ) else ( set PYCMD=py -3.11 )
)

if not exist "film\.venv\Scripts\python.exe" (
    echo [film] creating film\.venv ...
    %PYCMD% -m venv film\.venv || ( echo [film] venv creation FAILED & exit /b 1 )
)
"film\.venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
echo [film] installing torch + torchvision (cu128) ...
REM torch AND torchvision from the cu128 index FIRST: spandrel needs torchvision, and a
REM torchvision taken from PyPI pulls a CPU-only torch over the CUDA one (seen on lot 0).
"film\.venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet || ( echo [film] torch install FAILED & exit /b 1 )
"film\.venv\Scripts\python.exe" -m pip install -r film\requirements.txt --quiet || ( echo [film] pip install FAILED & exit /b 1 )
"film\.venv\Scripts\python.exe" -c "import torch, sys; sys.exit(0 if '+cu' in torch.__version__ else 1)" || ( echo [film] torch is a CPU build: re-run install_film.bat & exit /b 1 )

if not exist "film\models" mkdir "film\models"
if not exist "film\models\%FILM_FILE%" (
    echo [film] downloading %FILM_FILE% %FILM_TAG% ^(138 MB, Apache 2.0, dajes/frame-interpolation-pytorch^) ...
    curl -fL --retry 4 -o "film\models\%FILM_FILE%" "%FILM_URL%" || ( echo [film] download FAILED & exit /b 1 )
)
for %%A in ("film\models\%FILM_FILE%") do if not "%%~zA"=="%FILM_SIZE%" (
    echo [film] %FILM_FILE% has the wrong size ^(%%~zA bytes, expected %FILM_SIZE%^): deleted, run again
    del "film\models\%FILM_FILE%"
    exit /b 1
)
"film\.venv\Scripts\python.exe" -c "import torch; print('[film] torch', torch.__version__, '- GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')"
echo [film] ready.
endlocal
