@echo off
rem ---------------------------------------------------------------------------
rem build_tool.bat -- build one of the small diagnostic/bench tools under tools\.
rem
rem   usage:  tools\build_tool.bat <tool.cpp> [extra .cpp ...]
rem
rem   The tool source may be given as a path (tools\diag\limb_canary.cpp) or just
rem   a name -- a bare name is looked up in tools\diag, tools\bench, tools\test.
rem   The executable lands in build_vs18\tools\<name>.exe so the source tree stays
rem   clean.  Set DEFS=/DXXX to add compile definitions.
rem
rem   Examples:
rem     tools\build_tool.bat tools\diag\limb_canary.cpp
rem     tools\build_tool.bat tools\diag\mont_canary.cpp
rem     tools\build_tool.bat tools\diag\canary_sqr.cpp src\cpu\simd_mont_ifma.cpp
rem     tools\build_tool.bat tools\diag\dump_tmp.cpp src\cpu\ecm_edwards_save.cpp
rem     tools\build_tool.bat tools\bench\mers_test.cpp src\cpu\simd_mont_ifma.cpp
rem ---------------------------------------------------------------------------
setlocal
if "%~1"=="" ( echo usage: build_tool.bat ^<tool.cpp^> [extra .cpp ...] & exit /b 2 )

set SAVED=%CD%
set ROOT=%~dp0..
pushd "%ROOT%" || exit /b 1
set ROOT=%CD%

set TOOL=%~1
if not exist "%TOOL%" (
    for %%D in (tools\diag tools\bench tools\test) do (
        if exist "%%D\%~1" set TOOL=%%D\%~1
    )
)
if not exist "%TOOL%" ( echo build_tool: source not found: %~1 & popd & exit /b 2 )
for %%F in ("%TOOL%") do set NAME=%%~nF

rem vcvars: BuildTools first (the Community path disappeared after a VS update)
set VCV=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCV%" set VCV=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCV%" ( echo build_tool: vcvars64.bat not found & popd & exit /b 1 )
echo using "%VCV%"
call "%VCV%" x64 >nul 2>&1
if errorlevel 1 ( echo VCVARS FAILED & popd & exit /b 1 )

if not exist "%ROOT%\build_vs18\tools" mkdir "%ROOT%\build_vs18\tools"

cl /nologo /O2 /utf-8 /arch:AVX512 /EHsc %DEFS% ^
   /I "%ROOT%\include" /I "%ROOT%\src" /I "%ROOT%\src\core" /I "%ROOT%\src\cpu" ^
   /I "%ROOT%\tools\diag" /I "%ROOT%\tools\bench" ^
   /I "%ROOT%\third_party\gmp-zen3\dist\include" ^
   "%ROOT%\%TOOL%" %2 %3 %4 %5 %6 ^
   /Fe:"%ROOT%\build_vs18\tools\%NAME%.exe" ^
   /link /LIBPATH:"%ROOT%\third_party\gmp-zen3\dist\lib" gmp.lib
set RC=%errorlevel%
if %RC%==0 echo built build_vs18\tools\%NAME%.exe
rem keep the GMP runtime next to the tool so it runs without PATH tweaks
if exist "%ROOT%\third_party\gmp-zen3\dist\bin\gmp-10.dll" copy /y "%ROOT%\third_party\gmp-zen3\dist\bin\gmp-10.dll" "%ROOT%\build_vs18\tools\" >nul
popd
exit /b %RC%
