@echo off
rem ============================================================
rem  Open a command prompt with the project venv already active.
rem  Inside it, python and pip just work. Type exit to leave.
rem ============================================================
set "ROOT=%~dp0"

if not exist "%ROOT%.venv\Scripts\python.exe" (
    echo [error] venv not found. Create it first: see README section 3.
    pause
    exit /b 1
)

set "PATH=%ROOT%.venv\Scripts;%PATH%"

echo ============================================================
echo  rm_yolo_assess environment ready
echo    python -^> "%ROOT%.venv\Scripts\python.exe"
echo  Try: python train.py --epochs 100 --imgsz 640 --batch 16 --device 0
echo  Type exit to leave.
echo ============================================================
echo.
cd /d "%ROOT%"
cmd /k
