@echo off
setlocal enabledelayedexpansion
title SatQuery AI - Launcher
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found. Run Setup.bat first.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"

echo ============================================================
echo  SatQuery AI
echo ============================================================
echo.

REM ---------------------------------------------------------------
REM Free the ports from any previous run that didn't shut down
REM cleanly, so this always starts fresh instead of failing with
REM "address already in use". Prints what it's doing rather than
REM silently killing things.
REM ---------------------------------------------------------------
for /f "tokens=5" %%P in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    echo Port 8000 was already in use ^(PID %%P^) - closing it first.
    taskkill /F /PID %%P 2>nul
)
for /f "tokens=5" %%P in ('netstat -aon ^| findstr ":5500 " ^| findstr "LISTENING"') do (
    echo Port 5500 was already in use ^(PID %%P^) - closing it first.
    taskkill /F /PID %%P 2>nul
)

echo Starting the backend (a new window will open - leave it running)...
start "SatQuery AI Backend" /D "%~dp0backend" cmd /k "python -m uvicorn api.main:app --host 127.0.0.1 --port 8000"

echo Waiting for the backend to finish starting...
timeout /t 4 /nobreak >nul

echo Starting the frontend (a new window will open - leave it running)...
start "SatQuery AI Frontend" /D "%~dp0frontend" cmd /k "python -m http.server 5500"

timeout /t 2 /nobreak >nul

echo Opening the app in your browser...
start "" "http://localhost:5500"

echo.
echo ============================================================
echo  Running.
echo ============================================================
echo Two other windows just opened - "SatQuery AI - Backend" and
echo "SatQuery AI - Frontend". Both need to stay open while you use
echo the app; closing either one stops that half of the program.
echo.
echo To stop everything: close those two windows (or this one -
echo either way, rerunning Start_program.bat later cleans up any
echo leftover process automatically before starting fresh).
echo ============================================================
pause
