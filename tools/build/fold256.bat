powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0local_build.ps1" -BuildDir "D:\code\MPA-OpenCl\build_cuda_fold_t256" -Tiers "all" -Extra "-DECM_MERS_FOLD=1 -DECM_TPB=256"
pause
