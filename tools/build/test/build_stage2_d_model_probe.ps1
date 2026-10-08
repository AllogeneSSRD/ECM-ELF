#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/d_model_probe',[ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$vcvars=(Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if(-not $vcvars){throw 'vcvars64.bat not found'}
New-Item -ItemType Directory -Force $Build | Out-Null
$exe=Join-Path $Build 'stage2_d_model_probe.exe'
$log=Join-Path $Build 'build.log'
$deps=@('tools/test/stage2_d_model_probe.cu','tools/bench/stage2_d_model.cuh',
    'tools/bench/ntt_poly_probe.cu','tools/bench/ntt_coop_outer.cuh',
    'tools/bench/ntt_goldilocks_reduce.cuh','tools/bench/ntt_goldilocks_ptx.cuh','tools/build/test/build_stage2_d_model_probe.ps1')
$sourceHashes=[ordered]@{}
foreach($dep in $deps){$sourceHashes[$dep]=(Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash}
$line="call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch -I third_party/gmp-zen3/dist/include -Xcompiler /wd4819 tools/test/stage2_d_model_probe.cu -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`" > `"$log`" 2>&1"
$timer=[Diagnostics.Stopwatch]::StartNew()
& cmd.exe /c $line
if($LASTEXITCODE -ne 0){Get-Content $log -Tail 30;throw 'D model probe build failed'}
$timer.Stop()
foreach($dep in $deps){if((Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash -ne $sourceHashes[$dep]){throw "Source changed during build: $dep"}}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
$manifest=[ordered]@{exe=(Resolve-Path $exe).Path;sha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash;
    architecture=$Arch;build_seconds=$timer.Elapsed.TotalSeconds;sources=$sourceHashes}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $Build 'manifest.json')
Write-Host "built $exe"
