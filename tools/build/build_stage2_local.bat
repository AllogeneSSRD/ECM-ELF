@echo off
setlocal
rem Production: fixed PTX, AddSubMask=1, OuterUnrollU=0, split compile=6.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_stage2_local.ps1" %*
set "BUILD_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %BUILD_EXIT%
