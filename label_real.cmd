@echo off
rem ============================================================
rem  Open labelImg with ALL output redirected to a log file,
rem  so a crash can be diagnosed even if the window closes.
rem
rem  Log: .tmp\labelimg.log
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
set "IMG=%ROOT%raw\real\images"
set "CLS=%ROOT%raw\real\classes.txt"
set "LBL=%ROOT%raw\real\labels"
set "LOG=%ROOT%.tmp\labelimg.log"

if not exist "%ROOT%.tmp" mkdir "%ROOT%.tmp"
if not exist "%LBL%\classes.txt" copy /y "%CLS%" "%LBL%\classes.txt" >nul

echo ============================================================
echo   labelImg  -  log file enabled
echo ============================================================
echo.
echo   Log: %LOG%
echo.
echo   Steps:
echo     1. Draw one box with the "Create RectBox" toolbar button
echo     2. If it crashes, come back and tell me
echo     3. Press any key here to close this window
echo.

echo ==== run at %DATE% %TIME% ==== >> "%LOG%"
rem 每次启动前确保 PyQt5 兼容补丁已打上（幂等，重复跑无副作用）
"%PY%" "%ROOT%tools\patch_labelimg.py" >> "%LOG%" 2>&1
"%PY%" "%SCRIPT%" "%IMG%" "%CLS%" "%LBL%" >> "%LOG%" 2>&1

echo.
echo ============================================================
echo   labelImg exited. Log written to:
echo   %LOG%
echo ============================================================
pause
