param([Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
$Build=(Resolve-Path -LiteralPath $Build).Path
$vcvars=(Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat'|Sort-Object FullName -Descending|Select-Object -First 1).FullName
$exe=Join-Path $Build 'file_lock_delete_pending.exe'
$command="cl /nologo /std:c++17 /EHsc /MD /O2 /utf-8 /DNOMINMAX /I src /I include tools/test/file_lock_delete_pending.cpp src/io/files.cpp /Fo`"$Build/`" /Fe`"$exe`""
$batch=Join-Path $Build 'build.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`ncall `"$vcvars`" >nul 2>&1`r`nif errorlevel 1 exit /b 1`r`n$command`r`nexit /b %errorlevel%`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Out-File (Join-Path $Build 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content (Join-Path $Build 'build.log') -Tail 20;throw 'Lock regression build failed'}
Write-Host "Built $exe"
