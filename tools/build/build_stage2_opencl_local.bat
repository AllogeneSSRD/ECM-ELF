@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_stage2_opencl_local.ps1" %*
set "ecm_exit=%errorlevel%"
if "%~1"=="" pause
exit /b %ecm_exit%
