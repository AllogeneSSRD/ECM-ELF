#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/point_mersenne_probe',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$vcvars=(Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if(-not $vcvars){throw 'vcvars64.bat not found'}
New-Item -ItemType Directory -Force $Build | Out-Null
$body=[IO.File]::ReadAllText((Join-Path $repo 'tools/bench/stage2_tree_gpu.cu'))
$begin=$body.IndexOf('/* (hi, lo) = x*y + z + c, exactly */')
$end=$body.IndexOf('/* r = 2p: the reference''s xdbl, in Montgomery images */')
if($begin -lt 0 -or $end -le $begin){throw 'Production arithmetic extraction failed'}
$reference=Join-Path $Build 'stage2_point_reference.cuh'
$referenceBody=$body.Substring($begin,$end-$begin)
# The reference is the production SOS/REDC body with its optional selector removed.
$referenceBody=$referenceBody.Replace('__device__ __constant__ int g_s2g_point_mersenne_bits=0;','').Replace('#include "stage2_point_mersenne.cuh"','')
$referenceBody=$referenceBody.Replace('    const int mersenne_bits=g_s2g_point_mersenne_bits;','').Replace('    if(mersenne_bits){s2g_mersenne_mont_reduce<NW>(r,t,n,nw,mersenne_bits,out);return;}','')
[IO.File]::WriteAllText((Join-Path $repo $reference),$referenceBody,[Text.UTF8Encoding]::new($false))
$exe=Join-Path $Build 'stage2_point_mersenne_probe.exe'
$log=Join-Path $Build 'build.log'
$line="call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch -Xptxas -v -I `"$Build`" -I third_party/gmp-zen3/dist/include -Xcompiler /utf-8 -Xcompiler /wd4819 tools/test/stage2_point_mersenne_probe.cu -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`" > `"$log`" 2>&1"
$watch=[Diagnostics.Stopwatch]::StartNew()
& cmd.exe /c $line
$code=$LASTEXITCODE;$watch.Stop()
if($code -ne 0){Get-Content $log -Tail 35;throw 'Point Mersenne probe build failed'}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
$deps=@('tools/test/stage2_point_mersenne_probe.cu','tools/bench/stage2_point_mersenne.cuh','tools/bench/stage2_tree_gpu.cu','tools/build/test/build_stage2_point_mersenne_probe.ps1')
$hashes=[ordered]@{};foreach($dep in $deps){$hashes[$dep]=(Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash}
[ordered]@{exe=(Resolve-Path $exe).Path;sha256=(Get-FileHash -LiteralPath $exe).Hash;architecture=$Arch;build_seconds=$watch.Elapsed.TotalSeconds;reference_sha256=(Get-FileHash -LiteralPath $reference).Hash;sources=$hashes} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $Build 'manifest.json')
Write-Host ("built {0} ({1:N1}s)" -f $exe,$watch.Elapsed.TotalSeconds)
