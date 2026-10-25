#Requires -Version 5.1
param([Parameter(Mandatory=$true)][string]$WorkerBuild,
      [string]$Build='build_cuda_cmake/stage2_experiment_adapter')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
. (Join-Path $PSScriptRoot '../internal/cuda_build_state.ps1')
$WorkerBuild=Resolve-EcmSourcePath $repo $WorkerBuild
$Build=Resolve-EcmSourcePath $repo $Build
$manifest=Get-Content (Join-Path $WorkerBuild 'build_manifest.json') -Raw|ConvertFrom-Json
if($manifest.engine -ne 'development'){throw 'Require an explicitly experimental worker'}
$cuda=Resolve-EcmCudaToolkit -Archs $manifest.architecture
$vc=Resolve-EcmVcEnvironment $cuda
$gmp=$manifest.gmp_root
$dirs=@((Join-Path $gmp 'include'),'include','src','src/ecm/backend/cuda')
$inventory=Get-EcmCudaInventory $repo
$set=Get-EcmStage2SourceSet $repo $inventory development $dirs
Assert-EcmBuildUnits $repo $WorkerBuild $manifest $set.Sources $dirs
$source=@($inventory.groups.STAGE2_DEVELOPMENT)[0]
$unit=$manifest.compile_units.$source
$object=$unit.entry.output
if(-not $object -or (Get-FileHash $object).Hash -ne $unit.object_sha256){throw 'Experimental object missing or changed'}
$adapter='tools/test/stage2_experiment_adapter.cpp'
$hashes=Get-EcmSourceHashes $repo (Get-EcmSourceClosure $repo @($adapter) $dirs)
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
$exe=Join-Path $Build 'probe.exe'
$command='"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" -std=c++17 -O3 -arch='+$manifest.architecture+
    ' -I "'+$repo+'/include" -I "'+$gmp+'/include" -Xcompiler /utf-8 '+
    $adapter+' "'+$object+'" -L "'+$gmp+'/lib" -lgmp -o "'+$exe+'"'
$batch=Join-Path $Build 'build.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
. (Join-Path $PSScriptRoot '../internal/cuda_timing.ps1')
Enable-EcmNvccBatchTiming $batch $Build
& cmd.exe /d /c $batch 2>&1|Out-File (Join-Path $Build 'build.log') -Encoding utf8
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 25;throw 'Experimental adapter failed'}
Assert-EcmHashes $hashes (Get-EcmSourceHashes $repo $hashes.Keys) 'Adapter source changed'
if((Get-FileHash $object).Hash -ne $unit.object_sha256){throw 'Experimental object changed'}
Copy-Item (Join-Path $gmp 'bin/gmp-10.dll') $Build -Force
[ordered]@{experimental=$true;worker=$WorkerBuild;object=$object;object_sha256=$unit.object_sha256;
    worker_manifest_sha256=(Get-FileHash (Join-Path $WorkerBuild 'build_manifest.json')).Hash;
    sources=$hashes;exe_sha256=(Get-FileHash $exe).Hash}|ConvertTo-Json -Depth 6|
    Set-Content (Join-Path $Build 'manifest.json') -Encoding utf8
Write-Host "Built experimental-object adapter: $exe"
