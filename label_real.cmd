@echo off
rem ============================================================
rem  Open labelImg to fix the pre-labels of real-scene frames.
rem  All output goes to .tmp\labelimg.log so crashes are diagnosable.
rem
rem  NOTE: this file must stay pure ASCII. cmd.exe decodes .cmd in the
rem  system code page; non-ASCII comments break script parsing.
rem ============================================================
setlocal enabledelayedexpansion

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
    echo [error] Put this file inside the rm_yolo_assess folder.
    pause
    exit /b 1
)

set "PY=%ROOT%.venv\Scripts\python.exe"
set "SCRIPT=%ROOT%.tools\labelImg-src\labelImg.py"
set "PATCH=%ROOT%tools\patch_labelimg.py"
set "IMG=%ROOT%raw\real\images"
set "CLS=%ROOT%raw\real\classes.txt"
set "LBL=%ROOT%raw\real\labels"
set "LOG=%ROOT%.tmp\labelimg.log"

if not exist "%ROOT%.tmp" mkdir "%ROOT%.tmp"
if not exist "%LBL%\classes.txt" copy /y "%CLS%" "%LBL%\classes.txt" >nul

if not exist "%SCRIPT%" (
    echo [error] labelImg source not found: %SCRIPT%
    pause
    exit /b 1
)

echo ============================================================
echo   labelImg  -  log enabled
echo ============================================================
echo.
echo   Log file: %LOG%
echo.
echo   1. Draw one box with the Create RectBox toolbar button
echo   2. If it crashes, tell me - the log has the reason
echo   3. Press any key here after labelImg closes
echo.

echo ==== run at %DATE% %TIME% ==== >> "%LOG%"
"%PY%" "%PATCH%" >> "%LOG%" 2>&1
"%PY%" "%SCRIPT%" "%IMG%" "%CLS%" "%LBL%" >> "%LOG%" 2>&1

echo.
echo ============================================================
echo   labelImg exited. Log:
echo   %LOG%
echo ============================================================
pause
