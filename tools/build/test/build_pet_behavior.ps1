param([string]$Build='data/experiments/gui_v2/pet_behavior')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
$vc=(Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat' | Select-Object -First 1).FullName
if(-not $vc){throw 'MSVC not found'}
$source=Join-Path $repo 'tools/test/gui_v2/pet_behavior_test.cpp'
$exe=Join-Path $Build 'pet_behavior_test.exe'
$script=Join-Path $Build 'build.cmd'
[IO.File]::WriteAllText($script,"@echo off`r`ncall `"$vc`" >nul 2>&1`r`nif errorlevel 1 exit /b 1`r`ncl /nologo /std:c++17 /EHsc /utf-8 `"$source`" /Fo:`"$Build/`" /Fe:`"$exe`"`r`nexit /b %errorlevel%`r`n")
& cmd /d /c $script *> (Join-Path $Build 'build.log')
if($LASTEXITCODE){throw 'Behavior build failed'}
& $exe | Tee-Object (Join-Path $Build 'result.log')
if($LASTEXITCODE){throw 'Behavior checks failed'}
