@echo off
setlocal
rem General defaults are maintained in the matching PowerShell entry point.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_stage1_local.ps1" %*
set "BUILD_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %BUILD_EXIT%
