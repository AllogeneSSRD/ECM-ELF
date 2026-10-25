#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/stage1_gpu_progress_test')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$vcvars=(Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat'|Select-Object -First 1).FullName
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
$exe=Join-Path $Build 'stage1_gpu_progress_test.exe';$script=Join-Path $Build 'build.cmd'
$line="@echo off`r`ncall `"$vcvars`" >nul 2>&1`r`nif errorlevel 1 exit /b 1`r`n"+
    "cl /nologo /O2 /std:c++17 /EHsc /MD /utf-8 /I include /I src tools/test/stage1_gpu_progress_test.cpp src/ecm/platform/stage1_progress.cpp /Fe:`"$exe`" /Fo:`"$Build/`"`r`nexit /b %errorlevel%`r`n"
[IO.File]::WriteAllText($script,$line,[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c "`"$script`"" *> (Join-Path $Build 'build.log')
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 20;throw 'GPU progress test build failed'}
Write-Output "Built $exe"
