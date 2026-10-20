#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/stage2_giant_point_probe',[string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. tools/build/internal/cuda_toolchain.ps1
$cuda=Resolve-EcmCudaToolkit -Archs $Arch
$vc=Resolve-EcmVcEnvironment $cuda
New-Item -ItemType Directory -Force $Build | Out-Null
$exe=Join-Path $Build 'stage2_giant_point_probe.exe'
$log=Join-Path $Build 'build.log'
$line="$($vc.Setup) && `"$($cuda.Nvcc)`" -ccbin `"$($vc.Compiler)`" --split-compile=1 -std=c++17 -O3 -arch=$Arch -Xptxas -v -I `"$Build`" -I include -I src/cuda -I third_party/gmp-zen3/dist/include -Xcompiler /utf-8 -Xcompiler /wd4819 tools/test/stage2_giant_point_probe.cu -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`" > `"$log`" 2>&1"
& cmd.exe /d /c $line
if($LASTEXITCODE -ne 0){Get-Content $log -Tail 30;throw 'Giant point probe build failed'}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
$hashes=[ordered]@{}
foreach($p in @('include/ecm/cuda/stage2_launch.h','include/ecm/cuda/launch.h','include/ecm/cuda/tiers.def','kernels/cuda/stage2/point.cuh','kernels/cuda/stage2/point_mersenne.cuh','kernels/cuda/stage2/giant_chain.cuh','tools/test/stage2_giant_point_probe.cu','tools/build/test/build_stage2_giant_point_probe.ps1')){$hashes[$p]=(Get-FileHash $p).Hash}
[ordered]@{sha256=(Get-FileHash $exe).Hash;sources=$hashes;arch=$Arch} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Build 'manifest.json') -Encoding UTF8
