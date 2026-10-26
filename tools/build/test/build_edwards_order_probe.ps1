#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/edwards_order_probe')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
$cuda=Resolve-EcmCudaToolkit -Archs 89
$vc=Resolve-EcmVcEnvironment $cuda
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
$exe=Join-Path $Build 'edwards_curve_order_probe.exe';$obj=Join-Path $Build 'edwards_curve_order_probe.obj'
$command='"'+$vc.Compiler+'" /nologo /O2 /MD /std:c++17 /EHsc /utf-8 /I include /I src /I third_party/gmp-zen3/dist/include tools/test/edwards_curve_order_probe.cpp /Fo"'+$obj+'" /Fe"'+$exe+'" /link /LIBPATH:third_party/gmp-zen3/dist/lib gmp.lib'
$file=Join-Path $Build 'build.cmd'
[IO.File]::WriteAllText($file,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $file 2>&1 | Out-File (Join-Path $Build 'build.log') -Encoding utf8
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 30;throw 'compile failed'}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
$hashes=[ordered]@{}
foreach($source in @('tools/test/edwards_curve_order_probe.cpp','src/ecm/curve/edwards.h','tools/build/test/build_edwards_order_probe.ps1')){
    $hashes[$source]=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
}
[ordered]@{exe=$exe;sha256=(Get-FileHash $exe).Hash;runtime_sha256=(Get-FileHash (Join-Path $Build 'gmp-10.dll')).Hash;sources=$hashes}|ConvertTo-Json -Depth 5|Set-Content (Join-Path $Build 'manifest.json') -Encoding utf8
Write-Host "Built $exe"
