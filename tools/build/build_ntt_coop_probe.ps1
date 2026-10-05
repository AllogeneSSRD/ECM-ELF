#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/ntt_coop_probe',[ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$vcvars=(Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if(-not $vcvars){throw 'vcvars64.bat not found'}
New-Item -ItemType Directory -Force $Build | Out-Null
$exe=Join-Path $Build 'ntt_coop_outer_probe.exe'
$log=Join-Path $Build 'build.log'
$deps=@('tools/test/ntt_coop_outer_probe.cu','tools/bench/ntt_poly_probe.cu','tools/bench/ntt_coop_outer.cuh',
    'tools/bench/ntt_goldilocks_reduce.cuh','tools/build/build_ntt_coop_probe.ps1')
$hashes=[ordered]@{};foreach($dep in $deps){$hashes[$dep]=(Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash}
$line="call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch -Xptxas -v -I third_party/gmp-zen3/dist/include -Xcompiler /wd4819 tools/test/ntt_coop_outer_probe.cu -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`" > `"$log`" 2>&1"
$watch=[Diagnostics.Stopwatch]::StartNew()
& cmd.exe /c $line
$code=$LASTEXITCODE;$watch.Stop()
if($code -ne 0){Get-Content $log -Tail 30;throw 'NTT cooperative probe build failed'}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
foreach($dep in $deps){if((Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash -ne $hashes[$dep]){throw "Source changed during build: $dep"}}
[ordered]@{exe=(Resolve-Path $exe).Path;sha256=(Get-FileHash -LiteralPath $exe).Hash;architecture=$Arch;
    build_seconds=$watch.Elapsed.TotalSeconds;toolkit=(& nvcc --version | Out-String).Trim();sources=$hashes} |
    ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $Build 'manifest.json')
Write-Host ("built {0} ({1:N1}s)" -f $exe,$watch.Elapsed.TotalSeconds)
