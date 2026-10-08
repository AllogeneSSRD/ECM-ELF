#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/ntt_outer_v_probe',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(Test-Path -LiteralPath $Build){throw 'Use a fresh build directory'}
$vcvars=(Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if(-not $vcvars){throw 'vcvars64.bat not found'}
New-Item -ItemType Directory $Build | Out-Null
$deps=@('tools/test/ntt_outer_v_probe.cu','tools/test/ntt_coop_outer_probe.cu',
    'tools/bench/ntt_poly_probe.cu','tools/bench/ntt_coop_outer.cuh',
    'tools/bench/ntt_goldilocks_reduce.cuh','tools/bench/ntt_goldilocks_ptx.cuh',
    'tools/bench/ntt_carry_partial.cuh','tools/build/build_ntt_outer_v_probe.ps1')
$hashes=[ordered]@{}
foreach($dep in $deps){
    $hashes[$dep]=(Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash.ToLower()
    $dest=Join-Path $Build ('sources/'+$dep)
    New-Item -ItemType Directory -Force (Split-Path -Parent $dest) | Out-Null
    Copy-Item -LiteralPath $dep -Destination $dest
}
$exe=Join-Path $Build 'ntt_outer_v_probe.exe';$log=Join-Path $Build 'build.log'
$src=Join-Path $Build 'sources/tools/test/ntt_outer_v_probe.cu'
$line="call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch -DNTT_GL_FIXED_MODE=3 -DNTT_OUTER_UNROLL_U=0 -Xptxas -v -I third_party/gmp-zen3/dist/include -Xcompiler /wd4819 `"$src`" -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`" > `"$log`" 2>&1"
$watch=[Diagnostics.Stopwatch]::StartNew();& cmd.exe /d /c $line;$code=$LASTEXITCODE;$watch.Stop()
if($code -ne 0){Get-Content $log -Tail 35;throw 'Outer V probe build failed'}
foreach($dep in $deps){
    if((Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash.ToLower() -ne $hashes[$dep]){throw "Source changed during build: $dep"}
    if((Get-FileHash -LiteralPath (Join-Path $Build ('sources/'+$dep)) -Algorithm SHA256).Hash.ToLower() -ne $hashes[$dep]){throw "Snapshot changed: $dep"}
}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build
[ordered]@{exe=$exe;sha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLower();
    architecture=$Arch;gl_fixed_mode=3;compiled_outer_u=0;build_seconds=$watch.Elapsed.TotalSeconds;
    sources=$hashes;toolkit=(& nvcc --version | Out-String).Trim()} |
    ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $Build 'manifest.json')
Write-Host ("built {0} ({1:N1}s)" -f $exe,$watch.Elapsed.TotalSeconds)
