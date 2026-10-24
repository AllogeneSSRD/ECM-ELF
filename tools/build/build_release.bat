@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_release.ps1" %*
set "ECM_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %ECM_EXIT%
