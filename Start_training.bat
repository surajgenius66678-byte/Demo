@echo off
setlocal enabledelayedexpansion
title SatQuery AI - Training
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found. Run Setup.bat first.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"

cd training

echo ============================================================
echo  SatQuery AI - Training
echo ============================================================
echo.

REM ---------------------------------------------------------------
REM 1. Auto-detect the GPU and pick the matching config profile
REM ---------------------------------------------------------------
echo Detecting your GPU...
set "CONFIG="
for /f "delims=" %%C in ('python select_config.py') do set "CONFIG=%%C"
if not defined CONFIG (
    echo ERROR: GPU detection did not return a config. Stopping.
    pause
    exit /b 1
)
echo Using: !CONFIG!
echo.

REM ---------------------------------------------------------------
REM 2. Prepare data + split (resumable - safe to rerun if interrupted)
REM ---------------------------------------------------------------
echo ------------------------------------------------------------
echo  [1/5] Preparing dataset (download + filter + quality checks)
echo ------------------------------------------------------------
python -m data_prep.prepare_dataset --config "!CONFIG!"
if errorlevel 1 (
    echo.
    echo Dataset preparation failed or refused to continue - see the message
    echo above ^(it explains exactly what to check^). Stopping before wasting
    echo any GPU time on bad data.
    pause
    exit /b 1
)

echo.
echo ------------------------------------------------------------
echo  [2/5] Splitting train/val/test
echo ------------------------------------------------------------
python -m data_prep.splits --config "!CONFIG!"
if errorlevel 1 (
    echo.
    echo Splitting failed or refused to continue - see the message above.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------
REM 3. Safety gate: a quick dry run before the real (long, GPU-hours)
REM    run - a few real steps, nothing saved, confirms the config +
REM    environment + GPU combination actually works.
REM ---------------------------------------------------------------
echo.
echo ------------------------------------------------------------
echo  [3/5] Dry run - a few real steps, nothing saved, to catch any
echo         problem before the real run instead of during it
echo ------------------------------------------------------------
python train.py --config "!CONFIG!" --dry-run
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  Dry run FAILED. Stopping here on purpose - continuing to a
    echo  real multi-hour run on top of this would waste GPU time.
    echo ============================================================
    echo Common fixes:
    echo   - CUDA out of memory: see README.md's "If you hit CUDA out
    echo     of memory" section
    echo   - a missing/misnamed package: rerun Setup.bat
    pause
    exit /b 1
)

echo.
echo Dry run passed. Starting the real training run now.
echo This can take a while - resumable if interrupted, so closing this
echo window and rerunning Start_training.bat later continues from the
echo last saved checkpoint instead of starting over.
echo.
pause

REM ---------------------------------------------------------------
REM 4. The real training run (resumable by default - see train.py)
REM ---------------------------------------------------------------
echo.
echo ------------------------------------------------------------
echo  [4/5] Training ^(real run^)
echo ------------------------------------------------------------
python train.py --config "!CONFIG!"
if errorlevel 1 (
    echo.
    echo Training stopped with an error - see the message above. Fix
    echo whatever it points at and just run Start_training.bat again -
    echo it will resume from the last saved checkpoint automatically.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------
REM 5. Evaluate + export the checkpoint metadata Part 4 reads
REM ---------------------------------------------------------------
for /f "delims=" %%O in ('python -c "import yaml; print(yaml.safe_load(open('!CONFIG!'))['training']['output_dir'])"') do set "OUTPUT_DIR=%%O"
set "ADAPTER_DIR=!OUTPUT_DIR!\adapter"

echo.
echo ------------------------------------------------------------
echo  [5/5] Evaluating on the held-out test split, exporting metadata
echo ------------------------------------------------------------
python evaluate.py --config "!CONFIG!" --adapter-dir "!ADAPTER_DIR!"
if errorlevel 1 (
    echo Evaluation failed - see the message above.
    pause
    exit /b 1
)
python export_checkpoint.py --config "!CONFIG!" --adapter-dir "!ADAPTER_DIR!"
if errorlevel 1 (
    echo Export failed - see the message above.
    pause
    exit /b 1
)

for /f "delims=" %%M in ('python -c "import yaml; print(yaml.safe_load(open('!CONFIG!'))['paths']['checkpoint_metadata_path'])"') do set "META_PATH=%%M"

echo.
echo ============================================================
echo  Training complete.
echo ============================================================
echo Checkpoint metadata written to:
echo   !META_PATH!
echo.
echo To make Start_program.bat use this real, fine-tuned model instead
echo of Part 4's mock engine:
echo   1. Point backend\model_registry\config\models.yaml's checkpoint_path
echo      at the adapter directory above.
echo   2. In backend\api\main.py, change configure_engine^(use_mock=True^)
echo      to configure_engine^(use_mock=False^).
echo See README.md's "Wiring the trained checkpoint into the app" section.
echo ============================================================
pause
