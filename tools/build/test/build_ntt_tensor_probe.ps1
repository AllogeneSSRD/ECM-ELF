#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/ntt_tensor_probe',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$vcvars=(Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if(-not $vcvars){throw 'vcvars64.bat not found'}
New-Item -ItemType Directory -Force $Build | Out-Null
$exe=Join-Path $Build 'ntt_tensor_goldilocks_probe.exe'
$log=Join-Path $Build 'build.log'
$deps=@('tools/test/ntt_tensor_goldilocks_probe.cu','tools/experiments/cuda/ntt_tensor/goldilocks.cuh',
    'tools/experiments/cuda/ntt_variants/main.cu','tools/experiments/cuda/ntt_variants/ntt_coop_outer.cuh','tools/experiments/cuda/ntt_variants/ntt_goldilocks_reduce.cuh','tools/experiments/cuda/ntt_variants/ntt_goldilocks_ptx.cuh','tools/experiments/cuda/ntt_variants/ntt_carry_partial.cuh','tools/build/test/build_ntt_tensor_probe.ps1')
. (Join-Path $PSScriptRoot '../internal/cuda_build_state.ps1')
$deps=@(Get-EcmSourceClosure $repo $deps @('include','third_party/gmp-zen3/dist/include'))+
    @('tools/build/internal/cuda_build_state.ps1')
$hashes=[ordered]@{};foreach($dep in $deps){$hashes[$dep]=(Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash}
$hashes['src/ecm/backend/cuda/ntt/launch.h']=(Get-FileHash src/ecm/backend/cuda/ntt/launch.h).Hash
$hashes['src/ecm/backend/cuda/ntt/parameters.h']=(Get-FileHash src/ecm/backend/cuda/ntt/parameters.h).Hash
$hashes['src/ecm/backend/cuda/runtime/launch.h']=(Get-FileHash src/ecm/backend/cuda/runtime/launch.h).Hash
$line="call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch -Xptxas -v -I include -I third_party/gmp-zen3/dist/include -Xcompiler /utf-8 -Xcompiler /wd4819 tools/test/ntt_tensor_goldilocks_probe.cu -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`" > `"$log`" 2>&1"
$watch=[Diagnostics.Stopwatch]::StartNew()
. (Join-Path $PSScriptRoot '../internal/cuda_timing.ps1')
$line=ConvertTo-EcmTimedNvccCommand $line $Build
& cmd.exe /c $line
$code=$LASTEXITCODE;$watch.Stop()
if($code -ne 0){Get-Content $log -Tail 35;throw 'Goldilocks Tensor Core probe build failed'}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
[ordered]@{exe=(Resolve-Path $exe).Path;sha256=(Get-FileHash -LiteralPath $exe).Hash;
    architecture=$Arch;build_seconds=$watch.Elapsed.TotalSeconds;toolkit=(& nvcc --version | Out-String).Trim();
    sources=$hashes} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $Build 'manifest.json')
Write-Host ("built {0} ({1:N1}s)" -f $exe,$watch.Elapsed.TotalSeconds)
