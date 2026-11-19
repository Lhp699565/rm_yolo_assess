@echo off
rem ============================================================
rem  Open labelImg to fix the pre-labels of real-scene frames.
rem  Double-click this file.
rem
rem  Runs python in THIS window (no "start"), so if it crashes the
rem  error stays visible instead of the window flashing and closing.
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
set "EXE=%ROOT%.venv\Scripts\labelImg.exe"
set "SCRIPT=%ROOT%.tools\labelImg-src\labelImg.py"
set "IMG=%ROOT%raw\real\images"
set "CLS=%ROOT%raw\real\classes.txt"
set "LBL=%ROOT%raw\real\labels"

if not exist "%IMG%" (
    echo [error] Image folder not found: %IMG%
    pause
    exit /b 1
)
if not exist "%LBL%\classes.txt" (
    copy /y "%CLS%" "%LBL%\classes.txt" >nul
)

echo ============================================================
echo   Opening labelImg...
echo ============================================================
echo.
echo   IMPORTANT: click the "YOLO" button in the left toolbar.
echo   Shortcuts: W=draw  D/A=next/prev  Ctrl+S=save  Del=delete
echo.

if exist "%EXE%" (
    "%EXE%" "%IMG%" "%CLS%" "%LBL%"
) else if exist "%SCRIPT%" (
    "%PY%" "%SCRIPT%" "%IMG%" "%CLS%" "%LBL%"
) else (
    echo [error] labelImg not found.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   labelImg exited with code %ERRORLEVEL%
echo ============================================================
if not "%ERRORLEVEL%"=="0" (
    echo   If you see a traceback above, copy it and ask for help.
)
pause
