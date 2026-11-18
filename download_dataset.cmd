@echo off
rem ============================================================
rem  Double-click this file: download dataset, import it, check it.
rem  Safe to run again if the download breaks - it resumes.
rem ============================================================
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [error] venv not found: %PY%
    echo         See README section 3.
    pause
    exit /b 1
)

"%PY%" "%ROOT%tools\download_dataset.py" %*
echo.
echo ============================================================
echo  Done. Press any key to close this window.
echo ============================================================
pause >nul
