#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/ntt_outer_plan_probe',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot));Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
$cuda=Resolve-EcmCudaToolkit -Archs $Arch;$vc=Resolve-EcmVcEnvironment $cuda
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
New-Item -ItemType Directory -Force $Build | Out-Null
$exe=Join-Path $Build 'ntt_outer_plan_probe.exe';$log=Join-Path $Build 'build.log'
$deps=@('tools/bench/ntt_outer_plan_probe.cu','src/core/ecm_stage2_logging.h',
    'tools/build/test/build_ntt_outer_plan_probe.ps1','tools/build/internal/cuda_toolchain.ps1')
$deps+=@(Get-ChildItem src/cuda/stage2 -File -Filter 'ntt*.cuh' | Sort-Object Name | ForEach-Object {'src/cuda/stage2/'+$_.Name})
$hashes=[ordered]@{};foreach($dep in $deps){$hashes[$dep]=(Get-FileHash -LiteralPath $dep).Hash}
$line="$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n`"$($cuda.Nvcc)`" -ccbin `"$($vc.Compiler)`" -std=c++17 -O3 --default-stream per-thread -arch=$Arch -DNTT_GL_FIXED_MODE=3 -DNTT_GL_ADD_SUB_MASK=1 -Xptxas -v -I third_party/gmp-zen3/dist/include -Xcompiler /utf-8 -Xcompiler /wd4819 tools/bench/ntt_outer_plan_probe.cu -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`"`r`n"
$commandFile=Join-Path $Build 'compile.cmd';[IO.File]::WriteAllText($commandFile,"@echo off`r`n$line",[Text.UTF8Encoding]::new($false))
$watch=[Diagnostics.Stopwatch]::StartNew();& cmd.exe /d /c $commandFile 2>&1 | Out-File $log -Encoding utf8;$code=$LASTEXITCODE;$watch.Stop()
if($code -ne 0){Get-Content $log -Tail 40;throw 'Outer plan build failed'}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
foreach($dep in $deps){if((Get-FileHash -LiteralPath $dep).Hash -ne $hashes[$dep]){throw "Source changed: $dep"}}
[ordered]@{sha256=(Get-FileHash $exe).Hash;architecture=$Arch;default_stream='per-thread';build_seconds=$watch.Elapsed.TotalSeconds;
    toolkit=$cuda.VersionText;msvc=$vc.Version;gmp_dll_sha256=(Get-FileHash (Join-Path $Build 'gmp-10.dll')).Hash;sources=$hashes} |
    ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Build 'manifest.json') -Encoding UTF8
Write-Output "built $exe"
