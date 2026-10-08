@echo off
setlocal
rem Same arithmetic defaults as local, separate development output directory.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_stage1_dev.ps1" %*
set "BUILD_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %BUILD_EXIT%
