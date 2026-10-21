#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/stage1_checkpoint_probe_20261017')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/gmp_dependency.ps1')
$dependency=Resolve-EcmGmp -Repo $repo
$vcvars=(Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat'|Select-Object -First 1).FullName
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
New-Item -ItemType Directory -Force -Path $Build|Out-Null
$exe=Join-Path $Build 'stage1_checkpoint_probe.exe'
$script=Join-Path $Build 'build.cmd'
$line="@echo off`r`ncall `"$vcvars`" >nul 2>&1`r`nif errorlevel 1 exit /b 1`r`n"+
    "cl /nologo /O2 /std:c++17 /EHsc /MD /utf-8 /D NOMINMAX /I src /I include /I include /I src /I `"$($dependency.Root)/include`" tools/test/stage1_checkpoint_container_test.cpp src/ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.cpp src/ecm/platform/logging.cpp src/ecm/application/configuration/runtime.cpp /Fe:`"$exe`" /Fo:`"$Build/`" /link /LIBPATH:`"$($dependency.Root)/lib`" gmp.lib`r`nexit /b %errorlevel%`r`n"
[IO.File]::WriteAllText($script,$line,[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c "`"$script`"" *> (Join-Path $Build 'build.log')
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 25;throw 'checkpoint probe build failed'}
Copy-Item -LiteralPath (Join-Path $dependency.Root 'bin/gmp-10.dll') -Destination $Build -Force
Write-Host "Built checkpoint probe: $exe"
