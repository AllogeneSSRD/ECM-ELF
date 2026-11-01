@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0internal\build_gui_v2.ps1" %*
exit /b %errorlevel%
pause
