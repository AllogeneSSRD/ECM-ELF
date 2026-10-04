#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/d_model_probe',[string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$vcvars=(Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if(-not $vcvars){throw 'vcvars64.bat not found'}
New-Item -ItemType Directory -Force $Build | Out-Null
$exe=Join-Path $Build 'stage2_d_model_probe.exe'
$log=Join-Path $Build 'build.log'
$line="call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch -I third_party/gmp-zen3/dist/include -Xcompiler /wd4819 tools/test/stage2_d_model_probe.cu -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`" > `"$log`" 2>&1"
& cmd.exe /c $line
if($LASTEXITCODE -ne 0){Get-Content $log -Tail 30;throw 'D model probe build failed'}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
Write-Host "built $exe"
