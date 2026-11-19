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

echo ============================================================
echo   Opening labelImg...
echo ============================================================
echo.
echo   IMPORTANT: click the "YOLO" button in the left toolbar
echo   to switch save format. Otherwise it writes PascalVOC xml.
echo.
echo   Shortcuts:
echo     W          draw a new box
echo     D / A      next / previous image
echo     Ctrl + S   save current image
echo     Del        delete selected box
echo.

if exist "%EXE%" (
    echo   using: labelImg.exe
    start "" "%EXE%" "%IMG%" "%CLS%" "%LBL%"
    goto :done
)
if exist "%SCRIPT%" (
    echo   using: source script
    start "" "%PY%" "%SCRIPT%" "%IMG%" "%CLS%" "%LBL%"
    goto :done
)

echo [error] labelImg not found.
echo.
echo   .venv\Scripts\labelImg.exe          -- not found
echo   .tools\labelImg-src\labelImg.py     -- not found
echo.
echo   Ask for help before trying more installs.
pause
exit /b 1

:done
echo.
echo   Window launched. This console can be closed.
timeout /t 4 >nul
