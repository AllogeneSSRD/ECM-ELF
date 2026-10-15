#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/ntt_tensor_length_probe',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
$cuda=Resolve-EcmCudaToolkit -Archs $Arch
$vc=Resolve-EcmVcEnvironment $cuda
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
New-Item -ItemType Directory -Force $Build | Out-Null
$exe=Join-Path $Build 'ntt_tensor_length_probe.exe';$log=Join-Path $Build 'build.log'
$deps=@('tools/bench/ntt_tensor_length_probe.cu','tools/bench/ntt_tensor_goldilocks.cuh',
    'tools/test/ntt_tensor_goldilocks_probe.cu','src/core/ecm_stage2_logging.h',
    'tools/build/test/build_ntt_tensor_length_probe.ps1','tools/build/internal/cuda_toolchain.ps1')
$deps+=@(Get-ChildItem src/cuda/stage2 -File -Filter 'ntt*.cuh' | Sort-Object Name | ForEach-Object { 'src/cuda/stage2/'+$_.Name })
$hashes=[ordered]@{};foreach($dep in $deps){$hashes[$dep]=(Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash}
$line="$($vc.Setup) && `"$($cuda.Nvcc)`" -ccbin `"$($vc.Compiler)`" -std=c++17 -O3 --default-stream per-thread -arch=$Arch -DNTT_GL_FIXED_MODE=3 -DNTT_GL_ADD_SUB_MASK=1 -Xptxas -v -I third_party/gmp-zen3/dist/include -Xcompiler /utf-8 -Xcompiler /wd4819 tools/bench/ntt_tensor_length_probe.cu -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`" > `"$log`" 2>&1"
$commandFile=Join-Path $Build 'compile.cmd'
[IO.File]::WriteAllText($commandFile,"@echo off`r`n$line`r`n",[Text.UTF8Encoding]::new($false))
$watch=[Diagnostics.Stopwatch]::StartNew();& cmd.exe /c $commandFile;$code=$LASTEXITCODE;$watch.Stop()
if($code -ne 0){Get-Content $log -Tail 40;throw 'Tensor length probe build failed'}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
foreach($dep in $deps){if((Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash -ne $hashes[$dep]){throw "Source changed during build: $dep"}}
[ordered]@{exe=$exe;sha256=(Get-FileHash -LiteralPath $exe).Hash;architecture=$Arch;gl_fixed_mode=3;
    production_runtime=$true;default_stream='per-thread';build_seconds=$watch.Elapsed.TotalSeconds;toolkit=$cuda.VersionText;msvc=$vc.Version;
    gmp_dll_sha256=(Get-FileHash (Join-Path $Build 'gmp-10.dll')).Hash;sources=$hashes} |
    ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $Build 'manifest.json')
Write-Host ("built {0} ({1:N1}s)" -f $exe,$watch.Elapsed.TotalSeconds)
