#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/stage1_host_wait_probe',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
$cuda=Resolve-EcmCudaToolkit -Archs $Arch
$vc=Resolve-EcmVcEnvironment $cuda
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
New-Item -ItemType Directory -Path $Build -Force | Out-Null
$sources=@('tools/test/stage1_host_wait_probe.cu','kernels/cuda/cgbn_edwards_instances.cuh',
 'kernels/cuda/cgbn_edwards_point.cuh','kernels/cuda/cgbn_edwards_naf.cuh',
 'src/cuda/ecm_cuda_edwards_naf.inl','src/cuda/ecm_cuda_edwards_internal.h',
 'src/cuda/ecm_cuda_edwards_naf.h','src/cuda/ecm_cuda_edwards.h','src/cuda/ecm_cuda_edwards_tiers.inc',
 'src/core/ecm_edwards_curve.h','src/core/ecm_edwards_io.h','src/core/ecm_edwards_naf.h',
 'tools/build/test/build_stage1_host_wait_probe.ps1')+
 @(Get-ChildItem cgbn/include/cgbn -File -Recurse | ForEach-Object {$_.FullName})
$hashes=[ordered]@{}
foreach($source in $sources){$hashes[$source]=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash}
$exe=Join-Path $Build 'probe.exe'
$command='"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" -std=c++17 -O2 -arch='+$Arch+
 ' --pre-include=stdint.h -Xcompiler /MD -Xcompiler /utf-8 -Xcompiler /wd4819'+
 ' -I cgbn/include -I third_party/gmp-zen3/dist/include tools/test/stage1_host_wait_probe.cu'+
 ' -L third_party/gmp-zen3/dist/lib -lgmp -o "'+$exe+'"'
$batch=Join-Path $Build 'build.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1 | Out-File (Join-Path $Build 'build.log') -Encoding utf8
if($LASTEXITCODE -ne 0){throw 'Stage1 wait probe build failed; inspect build.log'}
Copy-Item -LiteralPath third_party/gmp-zen3/dist/bin/gmp-10.dll -Destination $Build -Force
foreach($source in $sources){if($hashes[$source] -ne (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash){throw "source changed during build: $source"}}
[ordered]@{diagnostic_only=$true;architecture=$Arch;cuda=$cuda.VersionText;compiler=$vc.Version;
 sources=$hashes;executable_sha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash;
 gmp_sha256=(Get-FileHash -LiteralPath (Join-Path $Build 'gmp-10.dll') -Algorithm SHA256).Hash
}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $Build 'manifest.json') -Encoding utf8
Write-Output "Diagnostic probe: $exe"
