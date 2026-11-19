@echo off
rem ============================================================
rem  Open labelImg to fix the pre-labels of real-scene frames.
rem  Double-click this file.
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

set "LB=%ROOT%.venv\Scripts\labelImg.exe"
if not exist "%LB%" (
    echo ============================================================
    echo   labelImg is not installed yet.
    echo ============================================================
    echo.
    echo   Copy this line into shell.cmd and press Enter:
    echo.
    echo   .venv\Scripts\python.exe -m pip install labelImg -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo.
    echo   Then double-click this file again.
    pause
    exit /b 1
)

set "IMG=%ROOT%raw\real\images"
set "CLS=%ROOT%raw\real\classes.txt"
set "LBL=%ROOT%raw\real\labels"

echo ============================================================
echo   Opening labelImg...
echo ============================================================
echo.
echo   IMPORTANT: click the "YOLO" button in the left toolbar
echo   to switch the save format. Otherwise it saves PascalVOC xml.
echo.
echo   Shortcuts:
echo     W          draw a new box
echo     D / A      next / previous image
echo     Ctrl + S   save current image
echo     Del        delete selected box
echo.
echo   Images : %IMG%
echo   Labels : %LBL%
echo.

start "" "%LB%" "%IMG%" "%CLS%" "%LBL%"
