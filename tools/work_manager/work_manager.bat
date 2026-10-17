@echo off
REM ============================================================================
REM  work_manager.bat - CMD wrapper for work_manager.ps1 (Windows port of
REM  gpu/work_manager.sh).
REM
REM  Usage: work_manager.bat [options] [WORK_FILE]
REM
REM    --run-log              enable one RUN_LOG file per task
REM    --sync-mode MODE       full | incremental (default incremental)
REM    --no-verbose           do not add -v to the ECM command line
REM    --pause                wait for a key press before closing the window
REM                           (useful when the script is started by a double click)
REM    -h, --help             show help
REM    other arguments        passed to work_manager.ps1 (e.g. -WorkFile x.txt)
REM
REM  Plain run:      work_manager.bat
REM  Full sync:      work_manager.bat --sync-mode full
REM ============================================================================

setlocal EnableExtensions
set "ROOT=%~dp0"
pushd "%ROOT%"

set "PSARGS="
set "PAUSE_AT_END="

:parse
if "%~1"=="" goto :run

if /i "%~1"=="--run-log"   ( set "PSARGS=%PSARGS% -RunLog" & shift & goto :parse )
if /i "%~1"=="--sync-mode" ( set "PSARGS=%PSARGS% -SyncMode %~2" & shift & shift & goto :parse )
if /i "%~1"=="--no-verbose" ( set "PSARGS=%PSARGS% -NoVerbose" & shift & goto :parse )
if /i "%~1"=="--pause"     ( set "PAUSE_AT_END=1" & shift & goto :parse )
if /i "%~1"=="--help"      ( set "PSARGS=%PSARGS% -Help" & shift & goto :parse )
if /i "%~1"=="-h"          ( set "PSARGS=%PSARGS% -Help" & shift & goto :parse )

set "PSARGS=%PSARGS% %~1"
shift
goto :parse

:run
echo.
echo ============================================
echo   ECM Work Manager
echo   work list : worktodo.txt         (default)
echo   exe       : .\ecm_cuda.exe       (default)
echo   verbose   : -v added by default  (--no-verbose to disable)
echo   sync mode : incremental          (default)
echo   log       : screen.log
echo ============================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%work_manager.ps1" %PSARGS%
set "EXITCODE=%ERRORLEVEL%"

popd

if defined PAUSE_AT_END (
    echo.
    echo Press any key to close this window . . .
    pause >nul
)

exit /b %EXITCODE%
