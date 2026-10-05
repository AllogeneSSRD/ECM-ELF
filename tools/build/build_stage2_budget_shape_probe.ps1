#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/stage2_budget_shape_probe',
      [string]$SourceRoot='build_cuda_cmake/_point_scratch_20261005/native_calibrated/sources',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop';$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot);Set-Location $repo
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
$SourceRoot=(Resolve-Path $SourceRoot).Path
if(Test-Path (Join-Path $Build 'manifest.json')){throw 'Use a fresh build directory'}
$vcvars=(Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue|Select-Object -First 1).FullName
if(-not $vcvars){throw 'vcvars64.bat not found'}
$deps=@('tools/bench/ntt_poly_probe.cu','tools/bench/ntt_coop_outer.cuh','tools/bench/ntt_goldilocks_reduce.cuh','tools/bench/ntt_goldilocks_ptx.cuh')
$sources=[ordered]@{};$generated=[ordered]@{}
foreach($dep in $deps){$src=Join-Path $SourceRoot $dep;$dest=Join-Path $Build ('_src/'+$dep);New-Item -ItemType Directory -Force (Split-Path -Parent $dest)|Out-Null;Copy-Item -LiteralPath $src -Destination $dest;$sources[$dep]=(Get-FileHash $src).Hash;$generated['_src/'+$dep]=(Get-FileHash $dest).Hash}
$fixture='tools/test/stage2_budget_shape_probe.cu';$dest=Join-Path $Build ('_src/'+$fixture);New-Item -ItemType Directory -Force (Split-Path -Parent $dest)|Out-Null;Copy-Item -LiteralPath $fixture -Destination $dest
$local=[ordered]@{};foreach($dep in @($fixture,'tools/build/build_stage2_budget_shape_probe.ps1')){$local[$dep]=(Get-FileHash $dep).Hash}
$generated['_src/'+$fixture]=(Get-FileHash $dest).Hash
$exe=Join-Path $Build 'stage2_budget_shape_probe.exe';$log=Join-Path $Build 'build.log'
$line="call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch -DNTT_GL_FIXED_MODE=3 -Xcompiler /wd4819 -Xcompiler /wd4996 -I third_party/gmp-zen3/dist/include `"$dest`" -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`" > `"$log`" 2>&1"
$watch=[Diagnostics.Stopwatch]::StartNew();& cmd.exe /c $line;$code=$LASTEXITCODE;$watch.Stop()
if($code -ne 0){Get-Content $log -Tail 25;throw 'Budget shape probe build failed'}
foreach($dep in $deps){if((Get-FileHash (Join-Path $SourceRoot $dep)).Hash -ne $sources[$dep]){throw "Source changed: $dep"}}
foreach($dep in $local.Keys){if((Get-FileHash $dep).Hash -ne $local[$dep]){throw "Local source changed: $dep"}}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
[ordered]@{exe=(Resolve-Path $exe).Path;sha256=(Get-FileHash $exe).Hash;source_root=$SourceRoot;sources=$sources;local_sources=$local;generated_sources=$generated;architecture=$Arch;gl_fixed_mode=3;build_seconds=$watch.Elapsed.TotalSeconds}|ConvertTo-Json -Depth 5|Set-Content -Encoding UTF8 (Join-Path $Build 'manifest.json')
Write-Host ("built {0:N1}s {1}" -f $watch.Elapsed.TotalSeconds,$exe)
