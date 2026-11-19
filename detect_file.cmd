@echo off
rem ============================================================
rem  Drag an image or video onto this file to run armor detection.
rem
rem  This file works from ANY folder depth inside the project:
rem  it walks up until it finds the ".venv" folder.
rem
rem  Usage:
rem    - Drag a .jpg/.png/.mp4 onto it
rem    - Or double-click and type/paste a path
rem ============================================================
setlocal enabledelayedexpansion

rem ---- find project root by walking up looking for .venv ----
set "D=%~dp0"
set "ROOT="
for /l %%i in (1,1,8) do (
    if exist "!D!.venv\Scripts\python.exe" (
        set "ROOT=!D!"
        goto :got_root
    )
    set "D=!D!..\"
)
:got_root

if not defined ROOT (
    echo [error] Could not find project root ^(folder containing .venv^).
    echo         Put this file somewhere inside rm_yolo_assess\ and try again.
    pause
    exit /b 1
)

set "PY=%ROOT%.venv\Scripts\python.exe"
set "OUT=%ROOT%runs\detect\quick"

rem ---- pick the model: prefer the one next to this file, else the baseline ----
set "W=%~dp0best.pt"
if not exist "%W%" set "W=%ROOT%runs\train\v030_baseline\weights\best.pt"
if not exist "%W%" (
    echo [error] No model ^(best.pt^) found.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo ============================================================
    echo  RM armor detector
    echo ============================================================
    echo.
    echo  Model : %W%
    echo.
    set /p "FILE=Drop a file here or type its path: "
) else (
    set "FILE=%~1"
)

if not defined FILE (
    echo No file given.
    pause
    exit /b 1
)

echo.
echo  Model : %W%
echo  Input : %FILE%
echo.

"%PY%" "%ROOT%detect.py" --weights "%W%" --source "%FILE%" --out "%OUT%" --show --conf 0.25

echo.
echo ============================================================
echo  Results saved to: runs\detect\quick
echo ============================================================
start "" "%OUT%"
pause
