@echo off
setlocal
rem Build/verify Stage2 components; build_release.bat creates complete packages.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_stage2_release.ps1" %*
set "BUILD_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %BUILD_EXIT%
