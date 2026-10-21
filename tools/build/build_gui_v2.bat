@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0dev\build_gui_v2.ps1" %*
exit /b %errorlevel%
