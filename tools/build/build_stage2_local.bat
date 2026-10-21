@echo off
setlocal
rem Defaults, including build_stage2_local output, live in the PowerShell entry.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_stage2_local.ps1" %*
set "BUILD_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %BUILD_EXIT%
