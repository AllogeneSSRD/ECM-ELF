powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0local_build.ps1" -BuildDir "D:\code\MPA-OpenCl\build_cuda_basic_t256" -Extra "-DECM_MERS_FOLD=0 -DECM_TPB=256"
pause
