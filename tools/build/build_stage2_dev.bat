@echo off
setlocal
rem Development engine, production arithmetic defaults, split compile=6.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_stage2_dev.ps1" %*
set "BUILD_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %BUILD_EXIT%
