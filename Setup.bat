@echo off
setlocal enabledelayedexpansion
title SatQuery AI - Setup
cd /d "%~dp0"

echo ============================================================
echo  SatQuery AI - Setup
echo ============================================================
echo This installs everything needed to run and train the system.
echo Safe to run more than once - it reuses what's already installed.
echo.

REM ---------------------------------------------------------------
REM 1. Find Python
REM ---------------------------------------------------------------
set "PYEXE="
python --version >nul 2>nul
if not errorlevel 1 set "PYEXE=python"
if not defined PYEXE (
    py --version >nul 2>nul
    if not errorlevel 1 set "PYEXE=py"
)
if not defined PYEXE (
    echo ERROR: Python was not found on this machine.
    echo.
    echo Install Python 3.10 or later from https://www.python.org/downloads/
    echo IMPORTANT: on the installer's first screen, tick the box that says
    echo "Add python.exe to PATH" - it's off by default.
    echo Then run this Setup.bat again.
    echo.
    pause
    exit /b 1
)
echo Found Python:
!PYEXE! --version
echo.

REM ---------------------------------------------------------------
REM 2. Create / reuse the virtual environment
REM ---------------------------------------------------------------
if exist ".venv\Scripts\activate.bat" (
    echo Virtual environment already exists in .venv - reusing it.
) else (
    echo Creating virtual environment in .venv ...
    !PYEXE! -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create the virtual environment. See the message above.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: could not activate the virtual environment.
    pause
    exit /b 1
)
echo.
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM ---------------------------------------------------------------
REM 3. Detect a GPU (best-effort - training still works without one
REM    detected here, e.g. if you'll train on an external/cloud GPU
REM    instead of this machine; see README.md)
REM ---------------------------------------------------------------
echo.
echo ------------------------------------------------------------
echo  Checking for an NVIDIA GPU on this machine...
echo ------------------------------------------------------------
set "HAS_GPU=0"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >nul 2>nul
if not errorlevel 1 (
    set "HAS_GPU=1"
    for /f "delims=" %%L in ('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader') do echo Found: %%L
) else (
    echo No NVIDIA GPU / driver detected on this machine via nvidia-smi.
    echo That's fine if you plan to train on an external or cloud GPU instead
    echo - see README.md's Training section for that path. Installing a
    echo CPU-only PyTorch build here either way; running training directly
    echo on this machine would need a real GPU to be practical.
)

REM ---------------------------------------------------------------
REM 4. Install PyTorch - the CUDA build if a GPU was found. Plain
REM    "pip install torch" on Windows installs a CPU-ONLY build by
REM    default (unlike Linux), so this has to be explicit.
REM ---------------------------------------------------------------
echo.
echo ------------------------------------------------------------
echo  Installing PyTorch...
echo ------------------------------------------------------------
if "!HAS_GPU!"=="1" (
    echo Installing the CUDA 12.1 build - compatible with RTX 20/30/40-series,
    echo A100, and most NVIDIA GPUs from the last several years.
    echo If this step fails, or you specifically need a different CUDA
    echo version, see https://pytorch.org/get-started/locally/ for the exact
    echo command for your setup, run it here manually, then rerun this script.
    pip install torch --index-url https://download.pytorch.org/whl/cu121
) else (
    pip install torch
)
if errorlevel 1 (
    echo ERROR: PyTorch install failed. See the message above.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------
REM 5. Install everything else
REM ---------------------------------------------------------------
echo.
echo ------------------------------------------------------------
echo  Installing the rest of requirements.txt...
echo ------------------------------------------------------------
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: some packages failed to install. See the message above.
    echo A common failure point is rasterio/GDAL - if that's what failed,
    echo see backend\preprocessing\README.md, or rasterio's own install
    echo docs at https://rasterio.readthedocs.io/en/stable/installation.html
    pause
    exit /b 1
)

REM ---------------------------------------------------------------
REM 6. Confirm what actually got installed
REM ---------------------------------------------------------------
echo.
echo ------------------------------------------------------------
echo  Confirming the install...
echo ------------------------------------------------------------
python -c "import torch; print('PyTorch', torch.__version__, '- CUDA available:', torch.cuda.is_available())"
if "!HAS_GPU!"=="1" (
    python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
    if errorlevel 1 (
        echo.
        echo WARNING: nvidia-smi found a GPU, but PyTorch does not see CUDA.
        echo Training will fall back to CPU, which is impractical for this
        echo model size. Double check the NVIDIA driver is installed and
        echo up to date, then rerun this script.
    )
)

echo.
echo ============================================================
echo  Setup complete.
echo ============================================================
echo   Start_program.bat   runs the app - upload images, ask questions
echo   Start_training.bat  fine-tunes the model - read README.md first
echo ============================================================
pause
