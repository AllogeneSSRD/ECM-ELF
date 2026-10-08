@echo off
setlocal
rem Parallel multi-arch build, automatic split, per-arch packages and ZIP archives.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_stage1_release.ps1" %*
set "BUILD_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %BUILD_EXIT%
