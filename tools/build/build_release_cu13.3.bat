@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_release.ps1" -Profile cu13.3 -Jobs 16 %*
set "ECM_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %ECM_EXIT%
