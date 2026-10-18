#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/relocation_0_36_0_candidate01/stage2')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
$manifest=Get-Content (Join-Path $Build 'build_manifest.json') -Raw | ConvertFrom-Json
if($manifest.engine -ne 'production'){throw 'Probe requires a completed production-engine test build'}
. (Join-Path $PSScriptRoot '../internal/gmp_dependency.ps1')
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
$dependency=Resolve-EcmGmp -Repo $repo
$cuda=Resolve-EcmCudaToolkit -Archs $manifest.architecture -CudaRoot $manifest.cuda_root
$vc=Resolve-EcmVcEnvironment $cuda
$engine=Join-Path $Build '_objects/ecm_cuda_stage2.obj'
$before=(Get-FileHash -LiteralPath $engine).Hash
$output=Join-Path $Build 'relocation_probe.exe'
$source=Join-Path $repo 'tools/test/stage2_relocation_probe.cpp'
$command="`"$($cuda.Nvcc)`" -ccbin `"$($vc.Compiler)`" -std=c++17 -O3 -arch=$($manifest.architecture) -Xcompiler=/MT " +
    "-I `"$repo/src/core`" -I `"$($dependency.Root)/include`" `"$source`" `"$engine`" -L `"$($dependency.Root)/lib`" -lgmp -o `"$output`""
$batch=Join-Path $Build '_objects/relocation_probe.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch
if($LASTEXITCODE -ne 0){throw 'Relocation probe build failed'}
if($before -ne (Get-FileHash -LiteralPath $engine).Hash){throw 'Engine object changed during probe build'}
[ordered]@{engine_object_sha256=$before;probe_sha256=(Get-FileHash $output).Hash;
    source_sha256=(Get-FileHash $source).Hash;engine_exe_sha256=$manifest.sha256} |
    ConvertTo-Json | Set-Content (Join-Path $Build 'relocation_probe_manifest.json') -Encoding UTF8
Write-Host "Produced $output"
