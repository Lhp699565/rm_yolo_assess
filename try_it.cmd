@echo off
rem ============================================================
rem  SIMPLEST DEMO - just double-click this file, nothing else.
rem  It runs detection on a built-in test image and opens the result.
rem ============================================================
setlocal enabledelayedexpansion

rem ---- find project root (folder containing .venv) ----
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
set "OUT=%ROOT%runs\detect\quick"

rem ---- pick a test image that has multiple targets in it ----
rem 000037 is verified to contain 10 labeled targets (9 detected).
set "IMG=%ROOT%datasets\armor\images\test\000037_view_targets.png"
if not exist "%IMG%" (
    rem fallback: first available test image
    for %%f in ("%ROOT%datasets\armor\images\test\*.png") do (
        if not exist "%IMG%" set "IMG=%%~ff"
    )
)

if not defined IMG (
    echo.
    echo [error] No test images found under datasets\armor\images\test
    echo         Run tools\download_dataset.py first.
    pause
    exit /b 1
)

echo ============================================================
echo   RM armor detector - quick demo
echo ============================================================
echo.
echo   Model : best.pt  (YOLOv8n, 6 classes, mAP50 = 0.97)
echo   Image : %IMG%
echo.
echo   Detecting... a window with boxes will pop up.
echo   Close that window to finish.
echo.

"%PY%" "%ROOT%detect.py" --weights "%ROOT%runs\train\v030_baseline\weights\best.pt" --source "%IMG%" --out "%OUT%" --show --conf 0.25

echo.
echo ============================================================
echo   Done. Opening result folder...
echo ============================================================
start "" "%OUT%"
echo.
echo   You can close this window now.
pause
