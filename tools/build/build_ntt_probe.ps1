#Requires -Version 5.1
<#
.SYNOPSIS
    Build tools/bench/ntt_poly_probe.cu (the integer-NTT Kronecker probe).

.DESCRIPTION
    Same idea as tools/build/build_stage2_probe.ps1, but the NTT probe is ONE translation
    unit with no dependency on the stage-2 kernels, so this is a plain nvcc compile + link.
    nvcc's output goes to <Build>/_nttprobe/ntt_poly_probe.log so a failure can be read
    without rerunning the whole build, and the object is skipped when it is newer than the
    source (use -Rebuild to force).

    The probe needs GMP (the per-coefficient reference product), so it links
    third_party/gmp-zen3/dist/lib and gmp-10.dll is copied next to the exe.

.PARAMETER Build
    Build directory (default build_cuda_cmake).

.PARAMETER Arch
    -arch for nvcc (default sm_89).

.PARAMETER Rebuild
    Recompile even when the object is newer than the source.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_ntt_probe.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_ntt_probe.ps1 -Rebuild
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

$src = 'tools/bench/ntt_poly_probe.cu'
$coop = 'tools/bench/ntt_coop_outer.cuh'
$goldReduce = 'tools/bench/ntt_goldilocks_reduce.cuh'
$goldPtx = 'tools/bench/ntt_goldilocks_ptx.cuh'
$carryCheck = 'tools/bench/ntt_carry_partial.cuh'
$inc = @(
    '-I third_party/gmp-zen3/dist/include'
) -join ' '
$gmpLib = 'third_party/gmp-zen3/dist/lib'
$gmpDll = 'third_party/gmp-zen3/dist/bin/gmp-10.dll'

$objDir = Join-Path $Build '_nttprobe'
$obj    = Join-Path $objDir 'ntt_poly_probe.obj'
$log    = Join-Path $objDir 'ntt_poly_probe.log'
$linkLog = Join-Path $objDir 'link.log'
$exe    = Join-Path $Build 'ntt_poly_probe.exe'
New-Item -ItemType Directory -Force $objDir | Out-Null

Write-Host ("ntt probe build: exe={0} arch={1}" -f $exe, $Arch)

$needCompile = $true
if (-not $Rebuild -and (Test-Path $obj) -and (Test-Path $exe)) {
    $objTime = (Get-Item $obj).LastWriteTime
    if ($objTime -gt (Get-Item $src).LastWriteTime -and $objTime -gt (Get-Item $coop).LastWriteTime -and $objTime -gt (Get-Item $goldReduce).LastWriteTime -and $objTime -gt (Get-Item $goldPtx).LastWriteTime -and $objTime -gt (Get-Item $carryCheck).LastWriteTime) { $needCompile = $false }
}

if ($needCompile) {
    # cmd /c (not Start-Process) so the file sandbox does not deny the child process; see
    # the note in build_stage2_probe.ps1.
    $line = "call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch $inc " +
            "-Xcompiler /wd4819 -c -o `"$obj`" `"$src`" > `"$log`" 2>&1"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & cmd.exe /c $line
    $code = $LASTEXITCODE
    $sw.Stop()
    Write-Host ("  {0,-22} {1,7:N1}s exit={2}" -f 'compile', $sw.Elapsed.TotalSeconds, $code)
    if ($code -ne 0) {
        Write-Host ("--- compile failed; tail of {0}" -f $log) -ForegroundColor Red
        if (Test-Path $log) { Get-Content $log -Tail 30 | ForEach-Object { Write-Host ("    " + $_) } }
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

# gmp-10.dll has to sit next to the exe (the same trap the other bench probes document).
if (Test-Path $gmpDll) { Copy-Item $gmpDll (Split-Path -Parent $exe) -Force }

Write-Host ("built {0} ({1:N1} MB)" -f $exe, ((Get-Item $exe).Length / 1MB))
exit 0
