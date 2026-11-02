@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0internal\build_gui_v2.ps1" %*
set "BUILD_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %BUILD_EXIT%
