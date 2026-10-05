#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/ntt_carry_partial_probe',[ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot);Set-Location $repo
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(Test-Path (Join-Path $Build 'manifest.json')){throw 'Use a fresh build directory'}
$vcvars=(Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if(-not $vcvars){throw 'vcvars64.bat not found'}
$deps=@('tools/bench/ntt_poly_probe.cu','tools/bench/ntt_carry_partial.cuh','tools/test/ntt_carry_partial_probe.cu','tools/build/build_ntt_carry_partial_probe.ps1')
$sources=[ordered]@{};$generated=[ordered]@{}
foreach($dep in $deps){$sources[$dep]=(Get-FileHash $dep).Hash}
foreach($dep in $deps[1..2]){$dest=Join-Path $Build ('_src/'+$dep);New-Item -ItemType Directory -Force (Split-Path -Parent $dest)|Out-Null;Copy-Item -LiteralPath $dep -Destination $dest;$generated['_src/'+$dep]=(Get-FileHash $dest).Hash}
$encoding=New-Object System.Text.UTF8Encoding($false)
$text=$encoding.GetString([IO.File]::ReadAllBytes((Join-Path $repo $deps[0])))
$begin=$text.IndexOf('__global__ void carry_residual_kernel(');$end=$text.IndexOf('/*'+"`r`n"+' * One PARALLEL',$begin)
if($end -lt 0){$end=$text.IndexOf('/*'+"`n"+' * One PARALLEL',$begin)}
if($begin -lt 0 -or $end -le $begin){throw 'Residual kernel extraction differs'}
$dest=Join-Path $Build '_src/tools/test/carry_reference.cuh';[IO.File]::WriteAllBytes($dest,$encoding.GetBytes($text.Substring($begin,$end-$begin)))
$generated['_src/tools/test/carry_reference.cuh']=(Get-FileHash $dest).Hash
$begin=$text.IndexOf('template <int ROUNDS>'+"`r`n"+'__device__ __forceinline__ unsigned long long carry_cone_value(')
if($begin -lt 0){$begin=$text.IndexOf('template <int ROUNDS>'+"`n"+'__device__ __forceinline__ unsigned long long carry_cone_value(')}
$end=$text.IndexOf('/* ---- twiddle tables',$begin)
$include=$text.IndexOf('#include "ntt_carry_partial.cuh"',$begin)
if($include -ge 0 -and $include -lt $end){$end=$include}
if($begin -lt 0 -or $end -le $begin){throw 'Cone extraction differs'}
$dest=Join-Path $Build '_src/tools/test/carry_cone_reference.cuh';[IO.File]::WriteAllBytes($dest,$encoding.GetBytes($text.Substring($begin,$end-$begin)))
$generated['_src/tools/test/carry_cone_reference.cuh']=(Get-FileHash $dest).Hash
$exe=Join-Path $Build 'ntt_carry_partial_probe.exe';$log=Join-Path $Build 'build.log';$src=Join-Path $Build '_src/tools/test/ntt_carry_partial_probe.cu'
$line="call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch -Xptxas -v `"$src`" -o `"$exe`" > `"$log`" 2>&1"
$watch=[Diagnostics.Stopwatch]::StartNew();& cmd.exe /c $line;$code=$LASTEXITCODE;$watch.Stop()
if($code -ne 0){Get-Content $log -Tail 30;throw 'Carry partial build failed'}
foreach($dep in $deps){if((Get-FileHash $dep).Hash -ne $sources[$dep]){throw "Source changed: $dep"}}
[ordered]@{exe=(Resolve-Path $exe).Path;sha256=(Get-FileHash $exe).Hash;architecture=$Arch;sources=$sources;generated_sources=$generated;build_seconds=$watch.Elapsed.TotalSeconds}|ConvertTo-Json -Depth 5|Set-Content -Encoding UTF8 (Join-Path $Build 'manifest.json')
Write-Host ("built {0:N1}s {1}" -f $watch.Elapsed.TotalSeconds,$exe)
