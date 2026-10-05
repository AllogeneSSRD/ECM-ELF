#Requires -Version 5.1
<#
.SYNOPSIS
    Build tools/bench/stage2_tree_gpu.cu -- the GPU polynomial layer + F product tree
    (M3 slice S1, docs/DEV_STAGE2_GPU_PLAN.md section 18).

.DESCRIPTION
    Same shape as tools/build/build_ntt_probe.ps1: ONE translation unit, a plain nvcc compile
    plus a link against the repo's GMP.  It is one TU because the tree deliberately calls the
    multiply that lives in tools/bench/ntt_poly_probe.cu -- it #includes that file with
    NTT_POLY_PROBE_NO_MAIN defined, so there is exactly one copy of every kernel, of the
    packing convention and of the exactness assertions (section 18 constraint 1: the tree must
    reuse the verified NTT, never re-implement it).

    nvcc's output goes to <Build>/_s2tree/stage2_tree_gpu.log so a failure can be read without
    rerunning the build; the object is skipped when it is newer than either source (use
    -Rebuild to force).  gmp-10.dll is copied next to the exe (the trap the other probes
    document).

.PARAMETER Build
    Build directory (default build_cuda_cmake).

.PARAMETER Arch
    -arch for nvcc (default sm_89).

.PARAMETER Rebuild
    Recompile even when the object is newer than the sources.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_stage2_tree_gpu.ps1
#>
param(
    [string]$Build = 'build_cuda_cmake',
    [string]$Arch = 'sm_89',
    [switch]$Rebuild
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo

$vcvars = (Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue |
           Select-Object -First 1).FullName
if (-not $vcvars) { throw "vcvars64.bat not found; install the MSVC build tools" }

$src  = 'tools/bench/stage2_tree_gpu.cu'
$dep  = 'tools/bench/ntt_poly_probe.cu'          # included by $src: a rebuild trigger too
$coop = 'tools/bench/ntt_coop_outer.cuh'
$dmodel = 'tools/bench/stage2_d_model.cuh'
$goldReduce = 'tools/bench/ntt_goldilocks_reduce.cuh'
$goldPtx = 'tools/bench/ntt_goldilocks_ptx.cuh'
$babyDevice = 'tools/bench/stage2_baby_device.cuh'
$babyHost = 'tools/bench/stage2_baby_host.cuh'
$pointMersenne = 'tools/bench/stage2_point_mersenne.cuh'
$carryCheck = 'tools/bench/ntt_carry_partial.cuh'
$inc  = '-I third_party/gmp-zen3/dist/include'
$gmpLib = 'third_party/gmp-zen3/dist/lib'
$gmpDll = 'third_party/gmp-zen3/dist/bin/gmp-10.dll'

$objDir = Join-Path $Build '_s2tree'
$obj    = Join-Path $objDir 'stage2_tree_gpu.obj'
$log    = Join-Path $objDir 'stage2_tree_gpu.log'
$linkLog = Join-Path $objDir 'link.log'
$exe    = Join-Path $Build 'stage2_tree_gpu.exe'
New-Item -ItemType Directory -Force $objDir | Out-Null

Write-Host ("stage2_tree_gpu build: exe={0} arch={1}" -f $exe, $Arch)

$needCompile = $true
if (-not $Rebuild -and (Test-Path $obj) -and (Test-Path $exe)) {
    $objTime = (Get-Item $obj).LastWriteTime
    if (-not (@($src,$dep,$coop,$dmodel,$goldReduce,$goldPtx,$babyDevice,$babyHost,$pointMersenne,$carryCheck) | Where-Object { (Get-Item $_).LastWriteTime -ge $objTime })) {
        $needCompile = $false
    }
}

if ($needCompile) {
    # cmd /c (not Start-Process) so the file sandbox does not deny the child process; see the
    # note in build_stage2_probe.ps1.
    $line = "call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch $inc " +
            "-Xcompiler /wd4819 -c -o `"$obj`" `"$src`" > `"$log`" 2>&1"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & cmd.exe /c $line
    $code = $LASTEXITCODE
    $sw.Stop()
    Write-Host ("  {0,-22} {1,7:N1}s exit={2}" -f 'compile', $sw.Elapsed.TotalSeconds, $code)
    if ($code -ne 0) {
        Write-Host ("--- compile failed; tail of {0}" -f $log) -ForegroundColor Red
        if (Test-Path $log) { Get-Content $log -Tail 40 | ForEach-Object { Write-Host ("    " + $_) } }
        throw "compile failed (exit $code)"
    }
} else {
    Write-Host "  up to date; link output reused"
}

$linkLine = "call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch " +
            "-o `"$exe`" `"$obj`" -L $gmpLib -lgmp > `"$linkLog`" 2>&1"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& cmd.exe /c $linkLine
$linkCode = $LASTEXITCODE
$sw.Stop()
if ($linkCode -ne 0) {
    Write-Host ("--- link failed; tail of {0}" -f $linkLog) -ForegroundColor Red
    if (Test-Path $linkLog) { Get-Content $linkLog -Tail 30 | ForEach-Object { Write-Host ("    " + $_) } }
    throw "link failed (exit $linkCode)"
}
Write-Host ("  {0,-22} {1,7:N1}s exit=0" -f 'link', $sw.Elapsed.TotalSeconds)

if (Test-Path $gmpDll) { Copy-Item $gmpDll (Split-Path -Parent $exe) -Force }

Write-Host ("built {0} ({1:N1} MB)" -f $exe, ((Get-Item $exe).Length / 1MB))
exit 0
