@echo off
setlocal
rem Explicit TPB A/B experiment. General build defaults live one directory above.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\build_stage1_local.ps1" -BuildDir build_cuda_basic_t256 -Extra "-DECM_MERS_FOLD=0 -DECM_TPB=256" %*
set "BUILD_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %BUILD_EXIT%
