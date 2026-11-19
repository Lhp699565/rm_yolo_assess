@echo off
rem ============================================================
rem  Drag an image or video onto this file to run armor detection.
rem  - Double-click: interactive prompt
rem  - Drag a file onto it: runs detection, shows result window,
rem    and opens the output folder
rem ============================================================
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "W=%ROOT%runs\train\v030_baseline\weights\best.pt"
set "OUT=%ROOT%runs\detect\quick"

if not exist "%PY%" (
    echo [error] venv not found: %PY%
    pause
    exit /b 1
)
if not exist "%W%" (
    echo [error] model not found: %W%
    pause
    exit /b 1
)

if "%~1"=="" (
    echo ============================================================
    echo  RM armor detector  -  drag an image or video onto me
    echo ============================================================
    echo.
    set /p "FILE=File path: "
) else (
    set "FILE=%~1"
)

if not defined FILE (
    echo No file given.
    pause
    exit /b 1
)

echo.
echo Running detection on:
echo   %FILE%
echo.

"%PY%" "%ROOT%detect.py" --weights "%W%" --source "%FILE%" --out "%OUT%" --show --conf 0.25

echo.
echo ============================================================
echo  Results saved to: runs\detect\quick
echo ============================================================
start "" "%OUT%"
pause
