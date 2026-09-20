@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
echo === cl version ===
cl 2>&1 | findstr /i "Version"
echo === nvcc compile sample_01_add ===
nvcc -O2 -arch=sm_89 --ptxas-options=-v -I cgbn/include -I "D:/code/vcpkg/installed/x64-windows/include" cgbn/samples/sample_01_add/add.cu -o build_cuda/smoke_add.exe "D:/code/vcpkg/installed/x64-windows/lib/gmp.lib"
echo === nvcc exit code: %errorlevel% ===
