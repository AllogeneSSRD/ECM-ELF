@echo off
setlocal
rem Mersenne-only fold experiment; requires suitable batch geometry.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\build_stage1_local.ps1" -BuildDir build_cuda_fold_t256 -Extra "-DECM_MERS_FOLD=1 -DECM_TPB=256" %*
set "BUILD_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %BUILD_EXIT%
