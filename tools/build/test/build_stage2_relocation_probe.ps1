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
. (Join-Path $PSScriptRoot '../internal/cuda_build_state.ps1')
$dependency=Resolve-EcmGmp -Repo $repo -Gmp $manifest.gmp_root
$cuda=Resolve-EcmCudaToolkit -Archs $manifest.architecture -CudaRoot $manifest.cuda_root
$vc=Resolve-EcmVcEnvironment $cuda
$inventory=Get-EcmCudaInventory $repo (Join-Path $Build 'cuda_sources.json')
$includeDirs=@((Join-Path $dependency.Root 'include'),'include','src','src/ecm/backend/cuda')
$sourceSet=Get-EcmStage2SourceSet $repo $inventory production $includeDirs
Assert-EcmBuildUnits $repo $Build $manifest $sourceSet.Sources $includeDirs -Production
$engineObjects=@()
$before=[ordered]@{}
foreach($engineSource in $inventory.groups.STAGE2_PRODUCTION) {
    $engine=Get-EcmObjectPath $manifest.compile_units.$engineSource.entry $Build
    $engineObjects+=$engine
    $before[$engineSource]=(Get-FileHash -LiteralPath $engine).Hash
}
$engineArgs=($engineObjects|ForEach-Object{'"'+$_+'"'}) -join ' '
$output=Join-Path $Build 'relocation_probe.exe'
$source=Join-Path $repo 'tools/test/stage2_relocation_probe.cpp'
$command="`"$($cuda.Nvcc)`" -ccbin `"$($vc.Compiler)`" -std=c++17 -O3 -arch=$($manifest.architecture) -Xcompiler=/MT " +
    "-I `"$repo/src`" -I `"$($dependency.Root)/include`" `"$source`" $engineArgs -L `"$($dependency.Root)/lib`" -lgmp -o `"$output`""
$batch=Join-Path $Build '_objects/relocation_probe.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch
if($LASTEXITCODE -ne 0){throw 'Relocation probe build failed'}
Assert-EcmBuildUnits $repo $Build $manifest $sourceSet.Sources $includeDirs -Production
[ordered]@{engine_objects=$before;probe_sha256=(Get-FileHash $output).Hash;
    source_sha256=(Get-FileHash $source).Hash;engine_exe_sha256=$manifest.sha256} |
    ConvertTo-Json | Set-Content (Join-Path $Build 'relocation_probe_manifest.json') -Encoding UTF8
Write-Host "Produced $output"
