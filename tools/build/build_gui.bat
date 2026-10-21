@echo off
setlocal
rem One GUI entry point; defaults are maintained in dev\build_gui.ps1.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0dev\build_gui.ps1" %*
set "BUILD_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %BUILD_EXIT%
