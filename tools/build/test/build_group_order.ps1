#Requires -Version 5.1
param([string]$Build='build_cpu_group_order',[string]$GmpRoot='third_party/gmp-zen3/dist')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
$GmpRoot=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($GmpRoot)
$vcvars=Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
if(-not $vcvars){throw 'MSVC x64 tools not found'}
if(-not (Test-Path -LiteralPath (Join-Path $GmpRoot 'lib/gmp.lib'))){throw 'GMP import library not found'}
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
$exe=Join-Path $Build 'group_order_probe.exe'
$command='cl.exe /nologo /O2 /MD /std:c++17 /EHsc /W4 /utf-8 /I src /I "'+$GmpRoot+'/include" src/ecm/core/group_order/group_order.cpp tools/test/group_order_probe.cpp /Fo"'+$Build+'/" /Fe"'+$exe+'" /link /LIBPATH:"'+$GmpRoot+'/lib" gmp.lib'
$file=Join-Path $Build 'build.cmd'
[IO.File]::WriteAllText($file,"@echo off`r`ncall `"$($vcvars.FullName)`" >nul 2>&1`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $file 2>&1 | Out-File (Join-Path $Build 'build.log') -Encoding utf8
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 50;throw 'CPU group order compilation failed'}
Copy-Item -LiteralPath (Join-Path $GmpRoot 'bin/gmp-10.dll') -Destination $Build -Force
$hashes=[ordered]@{}
foreach($source in @('src/ecm/core/group_order/group_order.h','src/ecm/core/group_order/group_order.cpp','src/ecm/core/curve/ecm_edwards_curve.h','tools/test/group_order_probe.cpp','tools/build/test/build_group_order.ps1')){
    $hashes[$source]=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
}
$compilerVersion=& cmd.exe /d /c ('call "'+$vcvars.FullName+'" >nul 2>&1 && cl.exe 2>&1') | Out-String
[ordered]@{exe=$exe;sha256=(Get-FileHash -LiteralPath $exe).Hash;runtime_sha256=(Get-FileHash -LiteralPath (Join-Path $Build 'gmp-10.dll')).Hash;sources=$hashes;compiler=$compilerVersion.Trim();flags='/O2 /MD /std:c++17';cuda_required=$false}|ConvertTo-Json -Depth 5|Set-Content (Join-Path $Build 'manifest.json') -Encoding utf8
Write-Host "Built $exe"
