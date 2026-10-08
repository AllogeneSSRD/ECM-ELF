#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/ntt_goldilocks_ptx_probe',[ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$vcvars=(Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if(-not $vcvars){throw 'vcvars64.bat not found'}
New-Item -ItemType Directory -Force $Build | Out-Null
$exe=Join-Path $Build 'ntt_goldilocks_ptx_probe.exe';$log=Join-Path $Build 'build.log'
$deps=@('tools/test/ntt_goldilocks_ptx_probe.cu','tools/bench/ntt_goldilocks_ptx.cuh',
    'tools/bench/ntt_goldilocks_reduce.cuh','tools/build/test/build_ntt_goldilocks_ptx_probe.ps1')
$hashes=[ordered]@{};foreach($dep in $deps){$hashes[$dep]=(Get-FileHash -LiteralPath $dep).Hash}
$line="call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch -Xptxas -v -I third_party/gmp-zen3/dist/include -Xcompiler /utf-8 -Xcompiler /wd4819 tools/test/ntt_goldilocks_ptx_probe.cu -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`" > `"$log`" 2>&1"
$watch=[Diagnostics.Stopwatch]::StartNew();& cmd.exe /c $line;$code=$LASTEXITCODE;$watch.Stop()
if($code -ne 0){Get-Content $log -Tail 30;throw 'Goldilocks PTX build failed'}
foreach($dep in $deps){if((Get-FileHash -LiteralPath $dep).Hash -ne $hashes[$dep]){throw "Source changed during build: $dep"}}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
[ordered]@{exe=(Resolve-Path $exe).Path;sha256=(Get-FileHash -LiteralPath $exe).Hash;architecture=$Arch;
    build_seconds=$watch.Elapsed.TotalSeconds;toolkit=(& nvcc --version | Out-String).Trim();sources=$hashes} |
    ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $Build 'manifest.json')
Write-Host ("built {0} ({1:N1}s)" -f $exe,$watch.Elapsed.TotalSeconds)
