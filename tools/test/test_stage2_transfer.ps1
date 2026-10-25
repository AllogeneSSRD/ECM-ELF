#Requires -Version 5.1
param([string]$Build='tools/test/_run/stage2_transfer')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$vc=(Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if(-not $vc){throw 'vcvars64.bat not found'}
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
New-Item -ItemType Directory -Force -Path $Build | Out-Null
[IO.File]::WriteAllText((Join-Path $Build ".gitignore"),"*`n")
$exe=Join-Path $Build 'stage2_transfer_test.exe'
$script=Join-Path $Build 'build.cmd'
$gmp=Join-Path $repo 'third_party/gmp-zen3/dist'
$line="@echo off`r`ncall `"$vc`" >nul 2>&1`r`nif errorlevel 1 exit /b 1`r`n"+
    "cl /nologo /std:c++17 /EHsc /MD /utf-8 /D NOMINMAX /I include /I src /I `"$gmp/include`" tools/test/stage2_transfer_test.cpp src/ecm/storage/transfer/ecm_stage2_transfer.cpp src/io/worktodo.cpp src/sync/files.cpp /Fe:`"$exe`" /Fo:`"$Build/`" /link /LIBPATH:`"$gmp/lib`" gmp.lib`r`nexit /b %errorlevel%`r`n"
[IO.File]::WriteAllText($script,$line,[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c "`"$script`"" *> (Join-Path $Build 'build.log')
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 25;throw 'Stage2 transfer fixture build failed'}
Copy-Item -LiteralPath (Join-Path $gmp 'bin/gmp-10.dll') -Destination $Build -Force
$run=Join-Path $Build ('run_'+[Guid]::NewGuid().ToString('N'))
& $exe $run
if($LASTEXITCODE -ne 0){throw 'Stage2 transfer fixture failed'}
