#Requires -Version 5.1
param([Parameter(Mandatory=$true)][string]$WorkerBuild,
      [string]$Build='build_cuda_cmake/cuda_p2_arithmetic_probe')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
. (Join-Path $PSScriptRoot '../internal/cuda_build_state.ps1')
$WorkerBuild=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkerBuild)
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
$manifest=Get-Content (Join-Path $WorkerBuild 'build_manifest.json') -Raw|ConvertFrom-Json
$cuda=Resolve-EcmCudaToolkit -Archs $manifest.architecture
$vc=Resolve-EcmVcEnvironment $cuda
$gmp=$manifest.gmp_root
$dirs=@((Join-Path $gmp 'include'),'include','src','src/ecm/backend/cuda')
$inventory=Get-EcmCudaInventory $repo
$sourceSet=Get-EcmStage2SourceSet $repo $inventory production $dirs
Assert-EcmBuildUnits $repo $WorkerBuild $manifest $sourceSet.Sources $dirs -Production
$unit=$manifest.compile_units.'src/ecm/backend/cuda/stage2/device/arithmetic_kernels.cu'
$object=$unit.entry.output
if(-not $object){$object=Join-Path $WorkerBuild '_objects/arithmetic_kernels.obj'}
$hash=(Get-FileHash $object).Hash
New-Item -ItemType Directory -Force $Build|Out-Null
$exe=Join-Path $Build 'probe.exe'
$command='"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" -std=c++17 -O3 -arch='+$manifest.architecture+
    ' -I "'+$repo+'/include" -I "'+$gmp+'/include" -Xcompiler /utf-8 '+
    ' tools/test/cuda_stage2_arithmetic_probe.cpp "'+$object+'" -L "'+$gmp+'/lib" -lgmp -o "'+$exe+'"'
$batch=Join-Path $Build 'build.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Out-File (Join-Path $Build 'build.log') -Encoding utf8
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 25;throw 'P2 arithmetic probe build failed'}
if((Get-FileHash $object).Hash -ne $hash){throw 'Production arithmetic object changed'}
Copy-Item (Join-Path $gmp 'bin/gmp-10.dll') $Build -Force
[ordered]@{object=$object;object_sha256=$hash;exe_sha256=(Get-FileHash $exe).Hash;worker=$WorkerBuild}|ConvertTo-Json|Set-Content (Join-Path $Build 'manifest.json') -Encoding utf8
Write-Host "Built production-object GMP probe: $exe"
