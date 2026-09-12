@echo off
setlocal enabledelayedexpansion
title SatQuery AI - Setup
cd /d "%~dp0"

REM ---------------------------------------------------------------
REM Logging: every run appends to setup_log.txt, and everything
REM printed to the screen is also saved there.
REM ---------------------------------------------------------------
set "LOGFILE=setup_log.txt"
echo. >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo  Setup run started: %DATE% %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

call :main >> "%LOGFILE%" 2>&1
set "RESULT=%ERRORLEVEL%"

if "%RESULT%"=="0" (
    echo.
    echo ============================================================
    echo  SETUP SUCCEEDED
    echo ============================================================
    echo   Start_program.bat   runs the app
    echo   Start_training.bat  fine-tunes the model - read README.md first
    echo   Full log saved to: %LOGFILE%
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo  SETUP FAILED  ^(exit code %RESULT%^)
    echo ============================================================
    echo   Check %LOGFILE% for the full details of what went wrong.
    echo   The last 20 lines are shown below:
    echo ------------------------------------------------------------
    powershell -Command "Get-Content '%LOGFILE%' -Tail 20"
    echo ------------------------------------------------------------
)
pause
exit /b %RESULT%


REM =================================================================
:main

echo ============================================================
echo  SatQuery AI - Setup
echo ============================================================
echo This installs everything needed to run and train the system.
echo Safe to run more than once - it reuses what's already installed.
echo.

REM ---------------------------------------------------------------
REM 0. Warn about other venv-like folders already sitting here.
REM    We never auto-adopt them (could belong to a different Python
REM    version or be set up differently) - just make sure the user
REM    knows they exist, so two environments don't quietly pile up.
REM ---------------------------------------------------------------
for %%D in (venv env venv312 venv311 venv310 .env) do (
    if exist "%%D\Scripts\activate.bat" (
        echo NOTE: found an existing "%%D" folder that looks like a
        echo virtual environment, separate from the ".venv" this script
        echo manages. It will be left untouched. If you meant to use it
        echo instead, activate it manually with:
        echo     %%D\Scripts\activate.bat
        echo.
    )
)

REM ---------------------------------------------------------------
REM 1. Find Python and confirm it's new enough (3.10+)
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
    exit /b 1
)

REM Detect the Windows Store "fake" python.exe stub. When no real Python
REM is installed, Windows puts a placeholder on PATH that answers to
REM "python" but isn't a real interpreter - running code through it does
REM nothing useful (or opens the Microsoft Store). A real interpreter can
REM always report sys.executable; the stub cannot.
!PYEXE! -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Found something called "python" on PATH, but it does not
    echo behave like a real Python interpreter. This usually means it's
    echo the Windows Store placeholder, not an actual install.
    echo.
    echo Fix: install real Python from https://www.python.org/downloads/
    echo then, in Windows Settings, under "Manage app execution aliases",
    echo turn OFF the "python.exe" / "python3.exe" entries for Microsoft
    echo Store, so they don't shadow the real one.
    exit /b 1
)

echo Found Python:
!PYEXE! --version

!PYEXE! -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.10 or later is required.
    echo Your version is shown above. Please upgrade Python and try again.
    exit /b 1
)
echo.

REM ---------------------------------------------------------------
REM 2. Create / reuse the virtual environment
REM ---------------------------------------------------------------
set "REBUILD_VENV=0"
if exist ".venv\Scripts\activate.bat" (
    ".venv\Scripts\python.exe" --version >nul 2>nul
    if errorlevel 1 (
        echo Existing .venv looks broken ^(its Python no longer runs^).
        echo Attempting to remove it and rebuild...
        rmdir /s /q ".venv"
        if exist ".venv" (
            echo ERROR: could not remove the broken .venv folder.
            echo This usually means it is locked by another program
            echo ^(antivirus, an open terminal with it activated, an
            echo editor/IDE indexing it, etc.^).
            echo.
            echo Close any programs using .venv, or delete the ".venv"
            echo folder manually, then run this script again.
            exit /b 1
        )
        set "REBUILD_VENV=1"
    ) else (
        echo Virtual environment already exists in .venv - reusing it.
    )
) else (
    set "REBUILD_VENV=1"
)

if "!REBUILD_VENV!"=="1" (
    echo Creating virtual environment in .venv ...
    !PYEXE! -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create the virtual environment. See the message above.
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: could not activate the virtual environment.
    exit /b 1
)
echo.
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo ERROR: failed to upgrade pip. Check your internet connection.
    exit /b 1
)

REM ---------------------------------------------------------------
REM 3. Detect a GPU
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
    echo CPU-only PyTorch build here either way.
)

REM ---------------------------------------------------------------
REM 4. Install PyTorch (skip if a matching build is already present)
REM ---------------------------------------------------------------
echo.
echo ------------------------------------------------------------
echo  Checking existing PyTorch install...
echo ------------------------------------------------------------
set "NEED_TORCH=1"
python -c "import torch" >nul 2>nul
if not errorlevel 1 (
    if "!HAS_GPU!"=="1" (
        python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
        if not errorlevel 1 set "NEED_TORCH=0"
    ) else (
        set "NEED_TORCH=0"
    )
)

if "!NEED_TORCH!"=="0" (
    echo PyTorch already installed and matches this machine - skipping.
) else (
    echo Installing PyTorch...
    if "!HAS_GPU!"=="1" (
        echo Installing the CUDA 12.1 build - compatible with RTX 20/30/40-series,
        echo A100, and most NVIDIA GPUs from the last several years.
        call :pip_with_retry "torch --index-url https://download.pytorch.org/whl/cu121"
    ) else (
        call :pip_with_retry "torch"
    )
    if errorlevel 1 (
        echo ERROR: PyTorch install failed after retrying. See the message above.
        exit /b 1
    )
)

REM ---------------------------------------------------------------
REM 5. Install requirements.txt (skip if unchanged since last run)
REM
REM    Hash is computed with Python's hashlib rather than certutil:
REM    certutil's text output format varies by Windows display
REM    language, which can silently break simple text-parsing.
REM    hashlib always behaves the same regardless of locale.
REM ---------------------------------------------------------------
echo.
echo ------------------------------------------------------------
echo  Installing the rest of requirements.txt...
echo ------------------------------------------------------------
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found in this folder.
    exit /b 1
)

set "HASH_FILE=.venv\requirements.hash"
set "OLD_HASH="
if exist "!HASH_FILE!" set /p OLD_HASH=<"!HASH_FILE!"

for /f "delims=" %%H in ('python -c "import hashlib; print(hashlib.sha256(open('requirements.txt','rb').read()).hexdigest())"') do set "NEW_HASH=%%H"

if "!NEW_HASH!"=="!OLD_HASH!" (
    echo requirements.txt unchanged since last successful install - skipping.
) else (
    call :pip_with_retry "-r requirements.txt"
    if errorlevel 1 (
        echo ERROR: some packages failed to install even after retrying.
        echo A common failure point is rasterio/GDAL - if that's what failed,
        echo see backend\preprocessing\README.md, or rasterio's own install
        echo docs at https://rasterio.readthedocs.io/en/stable/installation.html
        exit /b 1
    )
    echo !NEW_HASH! > "!HASH_FILE!"
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

exit /b 0


REM =================================================================
REM  Helper: pip_with_retry <args>
REM  Runs "pip install <args>". If it fails, waits 5 seconds and
REM  tries exactly once more before giving up - this rescues setup
REM  from a single dropped connection or a flaky mirror, without
REM  retrying forever on a real, permanent failure.
REM =================================================================
:pip_with_retry
set "PIP_ARGS=%~1"
pip install %PIP_ARGS%
if not errorlevel 1 exit /b 0

echo.
echo First attempt failed. Waiting 5 seconds and retrying once...
timeout /t 5 /nobreak >nul
pip install %PIP_ARGS%
if not errorlevel 1 exit /b 0

exit /b 1