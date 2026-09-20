@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
echo === nvcc -c cgbn_stage1.cu (dev build, no -D) ===
nvcc -c -O2 -arch=sm_89 --ptxas-options=-v -I kernels/cuda -I include -I cgbn/include -I cgbn/include/cgbn -I "D:/code/vcpkg/installed/x64-windows/include" kernels/cuda/cgbn_stage1.cu -o build_cuda/cgbn_stage1.obj 2>build_cuda/cu_err.txt
echo === nvcc exit code: %errorlevel% ===
findstr /i "error C1 C2 C3 C4005 undefined" build_cuda\cu_err.txt
echo === (only warnings if empty above) ===
