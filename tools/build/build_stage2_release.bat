@echo off
setlocal
rem Production arithmetic, parallel nvcc compilation and automatic ZIP packaging.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_stage2_release.ps1" %*
set "BUILD_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %BUILD_EXIT%
