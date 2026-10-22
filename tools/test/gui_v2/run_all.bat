@echo off
setlocal
rem Double-click to run. Pass --no-pause when calling from another script.
set "GUI_V2_NO_PAUSE="
for %%A in (%*) do if /I "%%~A"=="--no-pause" set "GUI_V2_NO_PAUSE=1"
where py.exe >nul 2>&1
if not errorlevel 1 (
    py.exe -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1
    if not errorlevel 1 goto run_py
)
where python.exe >nul 2>&1
if not errorlevel 1 (
    python.exe -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1
    if not errorlevel 1 goto run_python
)
echo ERROR: Python 3.11 or newer is required. Install Python and enable PATH.
set "GUI_V2_RESULT=2"
goto done

:run_py
py.exe -3 "%~dp0run_all.py" %*
set "GUI_V2_RESULT=%errorlevel%"
goto done

:run_python
python.exe "%~dp0run_all.py" %*
set "GUI_V2_RESULT=%errorlevel%"

:done
echo.
if not defined GUI_V2_NO_PAUSE pause
exit /b %GUI_V2_RESULT%
