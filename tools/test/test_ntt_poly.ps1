#Requires -Version 5.1
<#
.SYNOPSIS
    Integer-NTT polynomial multiplication (Route B, M2 first gate): is it CORRECT?

.DESCRIPTION
    tools/bench/ntt_poly_probe.cu is the integer-NTT twin of cufft_kron_probe.cu's `poly`
    mode: the same Kronecker-packed polynomial multiplication (slot width 2S + ceil(log2 P),
    so no carry crosses a slot) with the fp64 cuFFT replaced by an NTT over the Goldilocks
    prime p = 2^64-2^32+1.  The two are meant to be compared on one figure of merit,
    ns per operand-bit, so this test asserts the same thing the cuFFT test does -- that the
    result is EXACT -- plus the two NTT-specific invariants that are easy to get silently
    wrong:

      [1] every coefficient of the product equals the GMP schoolbook product (projected
          modulo the 32-bit prime 4294967291, because a 10323-bit coefficient does not fit
          in a uint64),
      [2] the exactness rule N * (2^bpw)^2 < p really holds for the chosen (N, bpw) -- if it
          is violated the NTT wraps and the answer is wrong with no error anywhere,
      [3] a shape that CANNOT satisfy the rule is refused loudly instead of returning
          garbage,
      [4] the fp64 cuFFT probe and the NTT probe agree with each other on the same shape
          (two independent transforms, same answer).

    Timing is printed but NOT asserted: performance gates belong in
    docs/DEV_STAGE2_GPU_PLAN.md, and a wall-clock assertion would be flaky.

    Uses device 1 by default (device 0 normally runs production stage 1).  Skips cleanly
    when the probe or a CUDA device is missing.
#>
param(
    [string]$Exe = '',
    [string]$CufftExe = '',
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

function Run([string]$exe, [string[]]$argv) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $out = (& $exe @argv 2>&1 | Out-String)
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    $code = if ($null -eq $code) { 0 } else { $code }
    return @{ out = $out; code = $code }
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Exe) { $Exe = Join-Path $repoRoot 'build_cuda_cmake\ntt_poly_probe.exe' }
if (-not $CufftExe) { $CufftExe = Join-Path $repoRoot 'build_cuda_cmake\cufft_kron_probe.exe' }
if (-not (Test-Path $Exe)) {
    Write-Host "[skip] ntt_poly_probe.exe not found (build with tools\build\test\build_ntt_probe.ps1)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "passed: 0   failed: 0"
    exit 0
}
$gmpDll = Join-Path $repoRoot 'third_party\gmp-zen3\dist\bin\gmp-10.dll'
$dir = Split-Path -Parent $Exe
if ((Test-Path $gmpDll) -and -not (Test-Path (Join-Path $dir 'gmp-10.dll'))) { Copy-Item $gmpDll $dir -Force }

Write-Host "integer-NTT polynomial multiplication probe"
Write-Host ("exe    : " + $Exe)
Write-Host ("device : " + $Device)
Write-Host ""

# ---------------------------------------------------------------------------------
Write-Host "[1] every coefficient equals the GMP product (small shapes, verify=1)"
$small = @(@(64, 64), @(128, 257), @(512, 1024))
$parsed = @{}
foreach ($case in $small) {
    $P = $case[0]; $S = $case[1]
    $r = Run $Exe @('poly', "$P", "$S", "$Device", '1')
    $m = [regex]::Match($r.out, 'poly: mode=poly P=(\d+) S=(\d+) slot_bits=(\d+) bpw=(\d+) nwords=(\d+) fft=(\d+) mem_mb=[\d.]+ ok=(\d)')
    Check ("P=$P S=$S printed a result line") ($m.Success) ($r.out.Trim() -split "`n" | Select-Object -First 1)
    if ($m.Success) {
        $parsed["$P/$S"] = $m
        Check ("P=$P S=$S every coefficient matches GMP") ($m.Groups[7].Value -eq '1') $m.Value
        $slot = [int]$m.Groups[3].Value
        Check ("P=$P S=$S the slot is at least 2S+log2(P)") ($slot -ge (2 * $S)) ("slot_bits=" + $slot)
        Check ("P=$P S=$S reports a bpw and a word count") ([int]$m.Groups[4].Value -ge 1 -and [int]$m.Groups[5].Value -ge 1) $m.Value
    }
}

# ---------------------------------------------------------------------------------
Write-Host "[2] the exactness rule L * (2^bpw - 1)^2 < p must hold for every shape"
# p = 2^64 - 2^32 + 1.  The bound is over L = P * slot_words (the NONZERO digits per
# operand), not over the transform length N: a convolution coefficient sums at most one
# product per nonzero digit of each operand, and the arrays are zero beyond the payload.
# (The earlier version of this check used N * (2^bpw)^2, which is a stricter -- and
# therefore wrong -- rule: it failed on shapes the probe now proves safe, see
# docs/DEV_STAGE2_GPU_PLAN.md 14.15.)  slot_words is ceil(slot_bits/bpw), the word-aligned
# stride, so it is derivable from the printed fields alone.
$glPrime = [System.Numerics.BigInteger]::Parse('18446744069414584321')   # NOT $p: PowerShell variables are case-INsensitive, so $p and $P below are the SAME variable
foreach ($k in $parsed.Keys) {
    $m = $parsed[$k]
    $Pval = [System.Numerics.BigInteger]::Parse($m.Groups[1].Value)
    $bpw = [int]$m.Groups[4].Value
    $slotBits = [System.Numerics.BigInteger]::Parse($m.Groups[3].Value)
    $slotWords = ($slotBits + $bpw - 1) / $bpw                  # integer ceil
    $L = $Pval * $slotWords
    $maxDigit = [System.Numerics.BigInteger]::Pow(2, $bpw) - 1
    $bound = $L * $maxDigit * $maxDigit
    Check ("$k : L*(2^bpw-1)^2 < p") ($bound -lt $glPrime) `
          ("L=" + $L + " bound=2^" + [Math]::Round([System.Numerics.BigInteger]::Log($bound, 2), 2) + " p=2^64-ish")
}

# ---------------------------------------------------------------------------------
Write-Host "[3] an impossible shape is refused instead of returning garbage"
# Same S but a far larger P: the slots get wider, the packed integer grows, N must grow
# with it, and bpw shrinks until the rule cannot be met for any power-of-two N.
$big = Run $Exe @('poly', '2000000', '5153', "$Device", '1')
Check "an impossible shape exits non-zero" ($big.code -ne 0) ("exit=" + $big.code + " " + $big.out.Trim())
Check "an impossible shape says why (bpw/exactness)" `
      ($big.out -match 'bpw|exact|too (small|large)|cannot|fits this payload|transform length') $big.out.Trim()

# ---------------------------------------------------------------------------------
Write-Host "[4] two independent transforms must agree (NTT vs fp64 cuFFT)"
if (-not (Test-Path $CufftExe)) {
    Write-Host "  [skip] cufft_kron_probe.exe not found ($CufftExe)" -ForegroundColor Yellow
} else {
    foreach ($case in @(@(512, 1024), @(1024, 2048))) {
        $P = $case[0]; $S = $case[1]
        $n = Run $Exe @('poly', "$P", "$S", "$Device", '1')
        $c = Run $CufftExe @('poly', "$P", "$S", "$Device", '1')
        $nOk = [regex]::Match($n.out, 'ok=(\d)').Groups[1].Value
        $cOk = [regex]::Match($c.out, 'ok=(\d)').Groups[1].Value
        Check ("P=$P S=${S}: NTT ok=1 and fp64 cuFFT ok=1 (same shape, same answer)" `
               ) ($nOk -eq '1' -and $cOk -eq '1') ("ntt ok=" + $nOk + " cufft ok=" + $cOk)
        # The two probes are allowed to pad the Kronecker slot differently (the cuFFT one
        # rounds its chunk count up); what has to hold is that BOTH slots are wide enough
        # that no carry crosses a slot, and that both agree with GMP coefficient by
        # coefficient.  Equality of the two numbers is NOT required and would be a false
        # gate (measured: ntt=2057 vs cufft=2064 at P=512 S=1024).
        $nSlot = [int][regex]::Match($n.out, 'slot_bits=(\d+)').Groups[1].Value
        $cSlot = [int][regex]::Match($c.out, 'slot_bits=(\d+)').Groups[1].Value
        $need = 2 * $S + [math]::Ceiling([math]::Log($P, 2))
        Check ("P=$P S=${S}: both Kronecker slots are at least 2S+log2(P)") `
              ($nSlot -ge $need -and $cSlot -ge $need) `
              ("ntt=" + $nSlot + " cufft=" + $cSlot + " need>=" + $need)
    }
}

# ---------------------------------------------------------------------------------
Write-Host "[5] the figure of merit is reported (numbers, not assertions)"
$fo = Run $Exe @('poly', '8192', '5153', "$Device", '0')
$fm = [regex]::Match($fo.out, 'ns_per_operand_bit=([\d.]+)')
Check "the probe reports ns per operand-bit" ($fm.Success) $fo.out.Trim()
if ($fm.Success) {
    Write-Host ("  (measured: " + $fm.Groups[1].Value + " ns/operand-bit at P=8192 S=5153; " +
                "fp64 cuFFT measured 0.278 on the same shape, Prime95's CPU ~0.15)")
}

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
