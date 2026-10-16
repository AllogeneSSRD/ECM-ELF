#Requires -Version 5.1
param([string]$CpuBuild='build_cpu_stage2',
      [string]$GpuBuild='build_cuda_cmake/edwards_gpu_workers/stage2',
      [switch]$GpuHostOnly)
$ErrorActionPreference='Stop'
& (Join-Path $PSScriptRoot 'build_cpu_stage2.ps1') -Build $CpuBuild
if($LASTEXITCODE -ne 0){throw 'CPU result-output build failed'}
$gpu=Join-Path $PSScriptRoot '../build_stage2_local.ps1'
& $gpu -Build $GpuBuild -Arch sm_89 -Engine production -HostOnly:$GpuHostOnly
if($LASTEXITCODE -ne 0){throw 'CUDA result-output build failed'}
