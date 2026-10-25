#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/edwards_save_ladder_check')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
$cuda=Resolve-EcmCudaToolkit -Archs 89
$vc=Resolve-EcmVcEnvironment $cuda
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
$command='"'+$vc.Compiler+'" /nologo /O2 /std:c++17 /EHsc /utf-8 /I include /I src /I third_party/gmp-zen3/dist/include tools/test/edwards_save_ladder_check.cpp /Fo"'+$Build+'/check.obj" /Fe"'+$Build+'/edwards_save_ladder_check.exe" /link /LIBPATH:third_party/gmp-zen3/dist/lib gmp.lib'
$file=Join-Path $Build 'build.cmd'
[IO.File]::WriteAllText($file,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $file 2>&1 | Out-File (Join-Path $Build 'build.log') -Encoding utf8
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 30;throw 'independent ladder build failed'}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
