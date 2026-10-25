#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/cufft_probe',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
. (Join-Path $PSScriptRoot '../internal/cuda_build_state.ps1')
$cuda=Resolve-EcmCudaToolkit -Archs $Arch
$vc=Resolve-EcmVcEnvironment $cuda
$Build=Resolve-EcmSourcePath $repo $Build
$gmp=Resolve-EcmSourcePath $repo 'third_party/gmp-zen3/dist'
$inventory=Get-EcmCudaInventory $repo
$sources=@($inventory.groups.EXPERIMENT_CUFFT)
if($sources.Count -ne 1){throw 'Require one explicit experimental cuFFT source'}
$dirs=@('include',(Join-Path $gmp 'include'))
$deps=@(Get-EcmSourceClosure $repo $sources $dirs)+@(
    'tools/build/test/build_cufft_probe.ps1','tools/build/internal/cuda_sources.cmake',
    'tools/build/internal/cuda_toolchain.ps1','tools/build/internal/cuda_build_state.ps1')
$hashes=Get-EcmSourceHashes $repo $deps
$libraries=@((Join-Path $cuda.Root 'lib/x64/cufft.lib'),(Join-Path $gmp 'lib/gmp.lib'))
$libraryHashes=[ordered]@{}
foreach($lib in $libraries){$libraryHashes[$lib]=(Get-FileHash -LiteralPath $lib).Hash}
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
$obj=Join-Path $Build 'cufft_probe.obj';$exe=Join-Path $Build 'cufft_kron_probe.exe'
$compile='"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" -std=c++17 -O3 -arch='+$Arch+
    ' -Xcompiler /MD -Xcompiler /utf-8 -I "'+$gmp+'/include" -c "'+$sources[0]+'" -o "'+$obj+'"'
$link='"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" -Xcompiler /MD "'+$obj+'" '+
    (($libraries|ForEach-Object {'"'+$_+'"'}) -join ' ')+' -o "'+$exe+'"'
$body="@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$compile`r`nif errorlevel 1 exit /b 1`r`n$link`r`nexit /b %errorlevel%`r`n"
$batch=Join-Path $Build 'build.cmd';$log=Join-Path $Build 'build.log'
[IO.File]::WriteAllText($batch,$body,[Text.UTF8Encoding]::new($false))
. (Join-Path $PSScriptRoot '../internal/cuda_timing.ps1')
Enable-EcmNvccBatchTiming $batch $Build
& cmd.exe /d /c $batch 2>&1|Out-File $log -Encoding utf8
if($LASTEXITCODE -ne 0){Get-Content $log -Tail 25;throw 'Experimental cuFFT probe build failed'}
Assert-EcmHashes $hashes (Get-EcmSourceHashes $repo $deps) 'cuFFT source changed during build'
foreach($lib in $libraries){if((Get-FileHash -LiteralPath $lib).Hash -ne $libraryHashes[$lib]){throw "Library changed: $lib"}}
Copy-Item (Join-Path $gmp 'bin/gmp-10.dll') $Build -Force
[ordered]@{experimental=$true;production_runtime=$false;architecture=$Arch;sources=$hashes;
    command=$body;cuda=$cuda.VersionText;compiler=$vc.Version;libraries=$libraryHashes;
    object_sha256=(Get-FileHash $obj).Hash;exe=$exe;sha256=(Get-FileHash $exe).Hash}|
    ConvertTo-Json -Depth 8|Set-Content (Join-Path $Build 'manifest.json') -Encoding utf8
Write-Host "Built experimental cuFFT probe: $exe"
