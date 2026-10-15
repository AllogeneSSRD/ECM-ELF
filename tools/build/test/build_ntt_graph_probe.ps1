#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/ntt_graph_probe',[string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot));Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
$cuda=Resolve-EcmCudaToolkit -Archs $Arch;$vc=Resolve-EcmVcEnvironment $cuda
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
New-Item -ItemType Directory -Force $Build | Out-Null
$deps=@('tools/test/ntt_graph_probe.cu','src/core/ecm_stage2_logging.h','tools/build/test/build_ntt_graph_probe.ps1','tools/build/internal/cuda_toolchain.ps1')
$deps+=@(Get-ChildItem src/cuda/stage2 -File -Filter 'ntt*.cuh' | Sort-Object Name | ForEach-Object {'src/cuda/stage2/'+$_.Name})
$hashes=[ordered]@{};foreach($dep in $deps){$hashes[$dep]=(Get-FileHash $dep).Hash}
$exe=Join-Path $Build 'ntt_graph_probe.exe';$log=Join-Path $Build 'build.log';$commandFile=Join-Path $Build 'compile.cmd'
$command="$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n`"$($cuda.Nvcc)`" -ccbin `"$($vc.Compiler)`" -std=c++17 -O3 -arch=$Arch -DNTT_GL_FIXED_MODE=3 -DNTT_GL_ADD_SUB_MASK=1 -I third_party/gmp-zen3/dist/include -Xcompiler /utf-8 -Xcompiler /wd4819 tools/test/ntt_graph_probe.cu -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`"`r`n"
[IO.File]::WriteAllText($commandFile,"@echo off`r`n$command",[Text.UTF8Encoding]::new($false))
$watch=[Diagnostics.Stopwatch]::StartNew();& cmd.exe /d /c $commandFile 2>&1 | Out-File $log -Encoding utf8;$code=$LASTEXITCODE;$watch.Stop()
if($code -ne 0){Get-Content $log -Tail 35;throw 'Graph probe build failed'}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
foreach($dep in $deps){if((Get-FileHash $dep).Hash -ne $hashes[$dep]){throw "Source changed: $dep"}}
[ordered]@{sha256=(Get-FileHash $exe).Hash;architecture=$Arch;default_stream='legacy';build_seconds=$watch.Elapsed.TotalSeconds;sources=$hashes} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Build 'manifest.json') -Encoding UTF8
Write-Output "built $exe"
