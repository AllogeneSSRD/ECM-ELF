#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/atomic_publish',[Parameter(Mandatory=$true)][string]$Evidence)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
$cuda=Resolve-EcmCudaToolkit -Archs 89
$vc=Resolve-EcmVcEnvironment $cuda
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
$Evidence=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Evidence)
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
$file=Join-Path $Build 'build.cmd'
$command='"'+$vc.Compiler+'" /nologo /O2 /std:c++17 /EHsc /utf-8 /I include /I src tools/test/atomic_publish_fixture.cpp src/ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.cpp /Fo"'+$Build+'/" /Fe"'+$Build+'/atomic_publish.exe"'
[IO.File]::WriteAllText($file,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $file 2>&1 | Out-File (Join-Path $Build 'build.log') -Encoding utf8
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 30;throw 'atomic publisher fixture build failed'}
& (Join-Path $Build 'atomic_publish.exe') $Evidence
if($LASTEXITCODE -ne 0){throw 'atomic publisher fixture failed'}
