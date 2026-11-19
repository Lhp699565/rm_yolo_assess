@echo off
rem ============================================================
rem  git wrapper - lets you run "git ..." inside this folder
rem  even though git is not on the system PATH.
rem
rem  It finds the git bundled with GitHub Desktop.
rem ============================================================
setlocal enabledelayedexpansion

set "GIT="
for /d %%d in ("%LOCALAPPDATA%\GitHubDesktop\app-*") do (
    if exist "%%~fd\resources\app\git\cmd\git.exe" (
        set "GIT=%%~fd\resources\app\git\cmd\git.exe"
    )
)
if not defined GIT (
    if exist "C:\Program Files\Git\cmd\git.exe" set "GIT=C:\Program Files\Git\cmd\git.exe"
)
if not defined GIT (
    if exist "%LOCALAPPDATA%\Programs\Git\cmd\git.exe" set "GIT=%LOCALAPPDATA%\Programs\Git\cmd\git.exe"
)

if not defined GIT (
    echo [error] git not found. Install Git for Windows or GitHub Desktop.
    exit /b 1
)

"%GIT%" %*
