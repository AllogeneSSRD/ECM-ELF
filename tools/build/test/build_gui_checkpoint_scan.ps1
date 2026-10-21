#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/gui_checkpoint_scan_20261017')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot));Set-Location $repo
$vcvars=(Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat'|Select-Object -First 1).FullName
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
New-Item -ItemType Directory -Force $Build|Out-Null
$exe=Join-Path $Build 'gui_checkpoint_scan.exe';$results=Join-Path $Build 'gui_save_names.exe';$script=Join-Path $Build 'build.cmd'
$line="@echo off`r`ncall `"$vcvars`" >nul 2>&1`r`nif errorlevel 1 exit /b 1`r`n"+
    "cl /nologo /O2 /std:c++17 /EHsc /MD /utf-8 /I src/ecm/application/gui tools/test/gui_checkpoint_scan_test.cpp src/ecm/application/gui/platform_win32.cpp /Fe:`"$exe`" /Fo:`"$Build/`" /link user32.lib gdi32.lib comdlg32.lib shell32.lib`r`nif errorlevel 1 exit /b 1`r`n"+
    "cl /nologo /O2 /std:c++17 /EHsc /MD /utf-8 /I src/ecm/application/gui tools/test/gui/results_test.cpp src/ecm/application/gui/results.cpp /Fe:`"$results`" /Fo:`"$Build/`"`r`nexit /b %errorlevel%`r`n"
[IO.File]::WriteAllText($script,$line,[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c "`"$script`"" *> (Join-Path $Build 'build.log')
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 20;throw 'GUI checkpoint scanner build failed'}
Write-Host "Built $exe"
Write-Host "Built $results"
