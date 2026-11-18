@echo off
rem ============================================================
rem  Run a python script with the project venv, no PATH setup needed.
rem
rem  Usage:
rem    run.cmd tools\prepare.py split --src raw --out datasets\armor
rem    run.cmd train.py --epochs 100 --imgsz 640 --batch 16 --device 0
rem ============================================================
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [error] venv not found: %PY%
    echo         Create it first: see README section 3.
    exit /b 1
)

"%PY%" %*
exit /b %ERRORLEVEL%
