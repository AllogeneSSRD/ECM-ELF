@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\code\MPA-OpenCl
"D:\code\vcpkg\downloads\tools\cmake-4.3.2-windows\cmake-4.3.2-windows-x86_64\bin\cmake.exe" --build build_cuda_cmake --target ecm_cuda 2>build_cuda\build_err.txt
echo === build exit: %errorlevel% ===
pause
