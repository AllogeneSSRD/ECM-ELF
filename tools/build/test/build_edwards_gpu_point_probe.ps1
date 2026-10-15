#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/edwards_gpu_point_probe',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$vcvars=(Get-ChildItem 'C:/Program Files*/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat' -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if(-not $vcvars){throw 'vcvars64.bat not found'}
New-Item -ItemType Directory -Force $Build | Out-Null
$exe=Join-Path $Build 'edwards_gpu_point_probe.exe'
$log=Join-Path $Build 'build.log'
$gpuObj=Join-Path $Build 'edwards_gpu_point_probe.obj'
$cpuObj=Join-Path $Build 'ecm_edwards_cpu.obj'
$prefix="call `"$vcvars`" >nul 2>&1 && "
$gpuCommand="nvcc -std=c++17 -O2 -arch=$Arch --pre-include=stdint.h -Xcompiler /utf-8 -Xcompiler /wd4819 "+
    "-I cgbn/include -I third_party/gmp-zen3/dist/include -I kernels/cuda -I src/cpu "+
    "-c tools/test/edwards_gpu_point_probe.cu -o `"$gpuObj`""
$cpuCommand="cl /nologo /O2 /std:c++17 /EHsc /utf-8 /I third_party/gmp-zen3/dist/include "+
    "/I src/cpu /c src/cpu/ecm_edwards_cpu.cpp /Fo:`"$cpuObj`""
$linkCommand="nvcc -arch=$Arch `"$gpuObj`" `"$cpuObj`" -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`""
'' | Set-Content -LiteralPath $log
foreach($command in @($gpuCommand,$cpuCommand,$linkCommand)){
    & cmd.exe /d /c ($prefix+$command) 2>&1 | Add-Content -LiteralPath $log
    if($LASTEXITCODE -ne 0){Get-Content $log -Tail 40;throw 'Edwards GPU point probe build failed'}
}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
$sources=@('kernels/cuda/cgbn_edwards_point.cuh','tools/test/edwards_gpu_point_probe.cu',
           'src/cpu/ecm_edwards_cpu.cpp','src/cpu/ecm_edwards_mont.h',
           'tools/build/test/build_edwards_gpu_point_probe.ps1')
$hashes=[ordered]@{}
foreach($source in $sources){$hashes[$source]=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash}
[ordered]@{exe=(Resolve-Path $exe).Path;sha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash;
           arch=$Arch;sources=$hashes} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $Build 'manifest.json')
Write-Host "Built $exe"
