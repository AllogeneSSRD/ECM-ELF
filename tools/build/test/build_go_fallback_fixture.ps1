#Requires -Version 5.1
param([string]$Build='build_go_fallback_fixture')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot));Set-Location $repo
$vc=(Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat'|Sort-Object FullName -Descending|Select-Object -First 1).FullName
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
New-Item -ItemType Directory -Force $Build|Out-Null
$command='cl.exe /nologo /O2 /MD /EHsc /std:c++17 /utf-8 /I src /I third_party/gmp-zen3/dist/include tools/test/go_fallback_fixture.cpp src/ecm/service/diagnostics/order_analysis.cpp src/ecm/core/group_order/group_order.cpp /Fo"'+$Build+'/" /Fe"'+$Build+'/go_fallback_fixture.exe" /link /LIBPATH:third_party/gmp-zen3/dist/lib gmp.lib'
[IO.File]::WriteAllText((Join-Path $Build 'build.cmd'),"@echo off`r`ncall `"$vc`" >nul 2>&1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
& cmd /d /c (Join-Path $Build 'build.cmd') 2>&1|Out-File (Join-Path $Build 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content (Join-Path $Build 'build.log') -Tail 25;throw 'fallback fixture compile failed'}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force

& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'build_diagnostics.ps1') -Build (Join-Path $Build 'diagnostics-build') -VcVars $vc
if($LASTEXITCODE){throw 'fixture diagnostics build failed'}
& python (Join-Path $repo 'tools/packaging/diagnostics_runtime.py') --package-build (Join-Path $Build 'diagnostics-build') --out $Build
if($LASTEXITCODE){throw 'fixture diagnostics deployment failed'}
