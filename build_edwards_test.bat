@echo off
rem Build the standalone Edwards stage-1 cross-validation unit (MSVC + vcpkg GMP).
setlocal

set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "GMP_ROOT=D:\code\vcpkg\installed\x64-windows"
set "SRC=D:\code\MPA-OpenCl\src\cpu\ecm_edwards_cpu.cpp"
set "OUT=D:\code\MPA-OpenCl\src\cpu\ecm_edwards_cpu.exe"

call "%VCVARS%" >nul || (echo vcvars64 failed & exit /b 1)

cl /nologo /O2 /EHsc /utf-8 /DBUILD_ECM_EDWARDS_STANDALONE /I "%GMP_ROOT%\include" /I "D:\code\MPA-OpenCl\src\cpu" "%SRC%" /Fe:"%OUT%" /link "%GMP_ROOT%\lib\gmp.lib"
if errorlevel 1 (echo COMPILE FAILED & exit /b 1)

copy /Y "%GMP_ROOT%\bin\gmp-10.dll" "D:\code\MPA-OpenCl\src\cpu\" >nul
echo OK: %OUT%
endlocal
