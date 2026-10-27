#Requires -Version 5.1
param([string]$Build='tools/test/_run/toml_config')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$vcvars=(Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if(-not $vcvars){throw 'vcvars64.bat not found'}
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
. (Join-Path $repo 'tools/build/internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
$exe=Join-Path $Build 'ecm_toml_fixture.exe'
$script=Join-Path $Build 'build.cmd'
$line="@echo off`r`ncall `"$vcvars`" >nul 2>&1`r`nif errorlevel 1 exit /b 1`r`n"+
    "cl /nologo /std:c++17 /EHsc /MD /utf-8 /D NOMINMAX /I include /I src tools/test/ecm_toml_fixture.cpp /Fe:`"$exe`" /Fo:`"$Build/`"`r`nexit /b %errorlevel%`r`n"
[IO.File]::WriteAllText($script,$line,[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c "`"$script`"" *> (Join-Path $Build 'build.log')
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 25;throw 'TOML fixture build failed'}
& $exe 'config/ecm.toml.default' (Join-Path $Build 'fixture.toml')
if($LASTEXITCODE -ne 0){throw 'TOML fixture failed'}
