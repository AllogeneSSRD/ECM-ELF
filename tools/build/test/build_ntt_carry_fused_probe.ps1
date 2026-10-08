#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/ntt_carry_fused_probe',[ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop';$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot));Set-Location $repo
if(Test-Path (Join-Path $Build 'manifest.json')){throw 'Use a fresh build directory'}
New-Item -ItemType Directory -Force $Build|Out-Null
$vcvars=(Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue|Select-Object -First 1).FullName
if(-not $vcvars){throw 'vcvars64.bat not found'}
$deps=@('tools/test/ntt_carry_fused_probe.cu','tools/bench/ntt_poly_probe.cu','tools/bench/ntt_coop_outer.cuh','tools/bench/ntt_goldilocks_reduce.cuh','tools/bench/ntt_goldilocks_ptx.cuh','tools/bench/ntt_carry_partial.cuh','tools/build/test/build_ntt_carry_fused_probe.ps1')
$hashes=[ordered]@{};foreach($dep in $deps){$hashes[$dep]=(Get-FileHash $dep).Hash}
$exe=Join-Path $Build 'ntt_carry_fused_probe.exe';$log=Join-Path $Build 'build.log'
$line="call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch -DNTT_GL_FIXED_MODE=3 -Xptxas -v -I third_party/gmp-zen3/dist/include -Xcompiler /wd4819 tools/test/ntt_carry_fused_probe.cu -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`" > `"$log`" 2>&1"
$watch=[Diagnostics.Stopwatch]::StartNew();& cmd.exe /c $line;$code=$LASTEXITCODE;$watch.Stop()
if($code -ne 0){Get-Content $log -Tail 30;throw 'Fused carry probe build failed'}
foreach($dep in $deps){if((Get-FileHash $dep).Hash -ne $hashes[$dep]){throw "Source changed: $dep"}}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
[ordered]@{exe=(Resolve-Path $exe).Path;sha256=(Get-FileHash $exe).Hash;architecture=$Arch;gl_fixed_mode=3;sources=$hashes;build_seconds=$watch.Elapsed.TotalSeconds}|ConvertTo-Json -Depth 5|Set-Content -Encoding UTF8 (Join-Path $Build 'manifest.json')
Write-Host ("built {0:N1}s {1}" -f $watch.Elapsed.TotalSeconds,$exe)
