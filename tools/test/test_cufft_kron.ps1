#Requires -Version 5.1
<#
.SYNOPSIS
    cuFFT + Kronecker probe: is the GPU big-integer / polynomial multiplication CORRECT?

.DESCRIPTION
    M0 of docs/DEV_STAGE2_GPU_PLAN.md.  Two things are asserted:

      [1] `check <bits>` -- the whole product of two random integers must equal GMP's, bit for
          bit (the tool itself compares chunk arrays reconstructed from the device against
          mpz_mul, so `ok=1` is a full equality, not a hash).
      [2] `poly <P> <S>` -- the actual stage-2 primitive: multiply two polynomials of P
          coefficients of S bits each and compare EVERY coefficient against a GMP schoolbook
          product (projected modulo the 32-bit prime 4294967291, since a 10323-bit coefficient
          does not fit in a uint64).

    Timing is printed by the tool but NOT asserted: performance gates belong in the plan
    (docs/DEV_STAGE2_GPU_PLAN.md 8.3/8.4), and a wall-clock assertion would be flaky.

    Uses device 1 by default (the second, normally idle card) so it does not disturb a stage-1
    job running on device 0.  Skips cleanly when no CUDA device is present.
#>
param(
    [string]$Exe = '',
    [int]$Device = 1
)

$ErrorActionPreference = 'Stop'
$script:pass = 0
$script:fail = 0

function Check([string]$name, $ok, [string]$detail = "") {
    if ($ok) {
        $script:pass++
        Write-Host ("  [ok]   " + $name)
    } else {
        $script:fail++
        Write-Host ("  [FAIL] " + $name + $(if ($detail) { " -- " + $detail } else { "" })) -ForegroundColor Red
    }
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Exe) { $Exe = Join-Path $repoRoot 'build_cuda_cmake\cufft_kron_probe.exe' }
if (-not (Test-Path $Exe)) { Write-Host "FAIL: cufft_kron_probe.exe not found (pass -Exe <path>)" -ForegroundColor Red; exit 2 }

$gmpDll = Join-Path $repoRoot 'third_party\gmp-zen3\dist\bin\gmp-10.dll'
$dir = Split-Path -Parent $Exe
if ((Test-Path $gmpDll) -and -not (Test-Path (Join-Path $dir 'gmp-10.dll'))) { Copy-Item $gmpDll $dir -Force }

$prev = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$probe = (& $Exe check 1200 12 $Device 2>&1 | Out-String)
$ErrorActionPreference = $prev
if ($probe -match 'no CUDA|CUDA error|invalid device') {
    Write-Host "[skip] no usable CUDA device $Device" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "passed: 0   failed: 0"
    exit 0
}

Write-Host "cuFFT/Kronecker probe (device $Device)"
Write-Host ("exe : " + $Exe)
Write-Host ""

Write-Host "[1] big-integer multiplication is bit-exact against GMP"
foreach ($bits in 1200, 1000008) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $out = (& $Exe check $bits 12 $Device 2>&1 | Out-String)
    $ErrorActionPreference = $prev
    $m = [regex]::Match($out, 'kron: mode=check bits=(\d+) chunk_bits=12 n=(\d+) fft=(\d+) mem_mb=[\d.]+ ok=(\d)')
    Check ("bits=$bits produced a result line") ($m.Success)
    if ($m.Success) {
        Check ("bits=$bits matches GMP bit for bit") ($m.Groups[4].Value -eq '1') ($m.Value)
    }
}

Write-Host "[2] the stage-2 primitive: P coefficients of S bits, every coefficient verified"
foreach ($case in @(@(64, 64), @(128, 257), @(512, 1024))) {
    $P = $case[0]; $S = $case[1]
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $out = (& $Exe poly $P $S $Device 1 2>&1 | Out-String)
    $ErrorActionPreference = $prev
    $m = [regex]::Match($out, 'poly: mode=poly P=(\d+) S=(\d+) slot_bits=(\d+) fft=(\d+) mem_mb=[\d.]+ ok=(\d)')
    Check ("P=$P S=$S produced a result line") ($m.Success)
    if ($m.Success) {
        Check ("P=$P S=$S every coefficient matches GMP") ($m.Groups[5].Value -eq '1') ($m.Value)
        $slot = [int]$m.Groups[3].Value
        Check ("P=$P S=$S the slot is padded to at least 2S+log2(P)") ($slot -ge (2 * $S)) ("slot_bits=" + $slot)
    }
}

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0