# ---------------------------------------------------------------------------
# test_cuda_mers_fold.ps1 -- Mersenne fold domain for the CUDA suyama family
# (ECM_MERS_FOLD build; see kernels/cuda/cgbn_stage1_kernel.h and
# docs/ECM_CGBN_OPTIMIZATION.md 9).
#
# For N = 2^k - 1 the fold replaces CGBN's Montgomery reduction (the Q*N half of
# mont_mul) with a fold of the 2*k-bit product.  The arithmetic is exact modular
# arithmetic either way, so the acceptance criterion is blunt: a fold run and a
# Montgomery run must produce the SAME stage-1 X for the same sigma -- and both
# must still agree with the CPU reference.
#
# Checks:
#   A. M991  param0: fold vs Montgomery build     -> 64/64 save lines identical
#   B. M991  param0: fold vs CPU (gmp backend)    -> 64/64 identical (CPU reference)
#   C. M991  param2: fold vs Montgomery build     -> 32/32 identical (const-diff path)
#   D. M3217 param0: fold vs Montgomery build     -> 64/64 identical (2nd tier)
#   E. guards: a non-Mersenne N and --gpu-param 3 must both be REFUSED by a fold build
#   F. throughput A/B on M4999 (tier 5120, 384 curves, B1=1e5): gputime + identical saves
#
# usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_cuda_mers_fold.ps1 `
#       [-CudaExe <montgomery build>] [-FoldExe <ECM_MERS_FOLD=1 build>] [-SkipSlow]
#
# The fold build must carry the tiers it is asked for: M991 -> 1024, M3217 -> 3328,
# M4999 -> 5120 (tools\build\build_stage1_local.ps1 -BuildDir build_cuda_fold `
#   -Tiers "1024,3328,4608,5120" -Extra "-DECM_MERS_FOLD=1")
# ---------------------------------------------------------------------------
param(
    [string]$CudaExe = 'build_cuda_cmake\ecm_cuda.exe',   # Montgomery reference build
    [string]$FoldExe = 'build_cuda_fold\ecm_cuda.exe',    # -DECM_MERS_FOLD=1 build
    [string]$CpuExe  = 'build_vs18\Release\ecm.exe',
    [int]$Device = 1,
    [int]$AbCurves = 768,        # throughput-A/B batch size (see section F)
    [switch]$SkipSlow,
    [switch]$ExpectFoldFaster   # gate on the A/B speedup (see section F); off by default
)

$ErrorActionPreference = 'Continue'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
function Resolve-Exe([string]$p) {
    if ([System.IO.Path]::IsPathRooted($p)) { return $p }
    return (Join-Path $root $p)
}
$CudaExe = Resolve-Exe $CudaExe
$FoldExe = Resolve-Exe $FoldExe
$CpuExe  = Resolve-Exe $CpuExe
foreach ($e in @($CudaExe, $FoldExe)) {
    if (-not (Test-Path $e)) { throw "missing executable: $e (build it first, see the header)" }
}

$work = Join-Path $root 'build_vs18\test_cuda_mers_fold'
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Path $work | Out-Null

$fails = 0
function Check([bool]$ok, [string]$what) {
    if ($ok) { Write-Host "  [PASS] $what" -ForegroundColor Green }
    else     { Write-Host "  [FAIL] $what" -ForegroundColor Red; $script:fails++ }
}
function Norm-Save([string]$f) {
    if ([string]::IsNullOrEmpty($f) -or -not (Test-Path $f)) { return @() }
    return Get-Content $f | Where-Object { $_ -match 'SIGMA=' } | ForEach-Object {
        (($_ -replace ' WHO=[^;]*;', '') -replace ' TIME=[^;]*;', '').Trim()
    }
}
function Compare-Saves([string]$a, [string]$b) {
    $na = Norm-Save $a; $nb = Norm-Save $b
    $n = [Math]::Min($na.Count, $nb.Count)
    $same = 0
    for ($i = 0; $i -lt $n; $i++) { if ($na[$i] -eq $nb[$i]) { $same++ } }
    return @{ a = $na.Count; b = $nb.Count; same = $same; diff = ($n - $same) }
}
function Write-N($file, $expr) {
    [System.IO.File]::WriteAllText($file, "$expr`n", ([System.Text.Encoding]::ASCII))
}
function Run-Tool([string]$exe, [string]$toolArgs, [string]$nfile, [string]$cwd) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ([string]::IsNullOrEmpty($cwd)) {
            $out = cmd /c "`"$exe`" $toolArgs < `"$nfile`"" 2>&1
        } else {
            $out = cmd /c "cd /d `"$cwd`" && `"$exe`" $toolArgs < `"$nfile`"" 2>&1
        }
    } finally { $ErrorActionPreference = $prev }
    return ($out | Out-String)
}
# gputime in ms as reported by the driver ("GPU stage1 returned: <ret> gputime=<ms> ms")
function Get-GpuTime([string]$output) {
    if ($output -match 'gputime=([0-9.]+)') { return [double]$Matches[1] }
    return -1.0
}

$n991   = Join-Path $work 'n991.txt'
$n3217  = Join-Path $work 'n3217.txt'
$n4999  = Join-Path $work 'n4999.txt'
$nnotm  = Join-Path $work 'n_not_mersenne.txt'
Write-N $n991  '(2^991-1)'
Write-N $n3217 '(2^3217-1)'
Write-N $n4999 '(2^4999-1)'
# 991 is odd, so 2^991+1 == 0 (mod 3): definitely NOT of the form 2^k-1
Write-N $nnotm '(2^991+1)'

Write-Host "=== A. M991 param0: fold build vs Montgomery build ==="
$gA = Join-Path $work 'A_fold.save'; $mA = Join-Path $work 'A_mont.save'
$oFold = Run-Tool $FoldExe "-gpu -d $Device --gpu-param 0 -sigma 100000 -gpucurves 64 -savea $gA 1e5 0" $n991 $work
Check ($oFold -match 'Mersenne fold domain') 'the fold build announces the fold domain (and reports k/t)'
$null = Run-Tool $CudaExe "-gpu -d $Device --gpu-param 0 -sigma 100000 -gpucurves 64 -savea $mA 1e5 0" $n991 $work
$r = Compare-Saves $gA $mA
Check (($r.a -eq 64) -and ($r.b -eq 64)) "both saves carry 64 curve lines (fold $($r.a), mont $($r.b))"
Check ($r.same -eq 64 -and $r.diff -eq 0) "fold and Montgomery agree on all 64 curves (sigma AND x)"

Write-Host ""
Write-Host "=== B. M991 param0: fold build vs CPU reference ==="
if (Test-Path $CpuExe) {
    $cB = Join-Path $work 'B_cpu'
    New-Item -ItemType Directory -Force $cB | Out-Null
    $null = Run-Tool $CpuExe "--method mont --backend gmp -sigma 100000 -gpucurves 64 --tmp-dir `"$cB`" --ckpt 0 1e5 0" $n991 $work
    $cBfile = (Get-ChildItem $cB -Filter *.save | Select-Object -First 1).FullName
    $r = Compare-Saves $gA $cBfile
    Check ($r.same -eq 64 -and $r.diff -eq 0) "the FOLD result matches the CPU reference on all 64 curves"
} else {
    Write-Host "  [SKIP] CPU exe not found at $CpuExe" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== C. M991 param2 (const-diff path): fold vs Montgomery ==="
# NOTE the sigma prefix must match the parametrization: "-sigma 3:..." selects
# -param 3 and the driver then refuses the conflict with --gpu-param 2.
$gC = Join-Path $work 'C_fold.save'; $mC = Join-Path $work 'C_mont.save'
$oC = Run-Tool $FoldExe "-gpu -d $Device --gpu-param 2 -sigma 2:777777 -gpucurves 32 -savea $gC 1e5 0" $n991 $work
Check ($oC -match 'param2 stage-1 x') 'the fold build runs the param2 family'
$null = Run-Tool $CudaExe "-gpu -d $Device --gpu-param 2 -sigma 2:777777 -gpucurves 32 -savea $mC 1e5 0" $n991 $work
$r = Compare-Saves $gC $mC
Check (($r.a -eq 32) -and ($r.same -eq 32) -and ($r.diff -eq 0)) "param2 fold == param2 Montgomery (32/32 lines)"

Write-Host ""
Write-Host "=== D. M3217 (tier 3328) param0: fold vs Montgomery ==="
$gD = Join-Path $work 'D_fold.save'; $mD = Join-Path $work 'D_mont.save'
$oFoldD = Run-Tool $FoldExe "-gpu -d $Device --gpu-param 0 -sigma 9007199254740847 -gpucurves 64 -savea $gD 1e5 0" $n3217 $work
Check ($oFoldD -match 'N = 2\^3217 - 1') 'the fold banner reports k=3217 for M3217'
$null = Run-Tool $CudaExe "-gpu -d $Device --gpu-param 0 -sigma 9007199254740847 -gpucurves 64 -savea $mD 1e5 0" $n3217 $work
$r = Compare-Saves $gD $mD
Check (($r.a -eq 64) -and ($r.same -eq 64) -and ($r.diff -eq 0)) "M3217: fold == Montgomery (64/64 lines)"

Write-Host ""
Write-Host "=== E. guards: a fold build must refuse non-Mersenne N and param3 ==="
$o = Run-Tool $FoldExe "-gpu -d $Device --gpu-param 0 -sigma 100000 -gpucurves 8 -savea $work\E.save 1e4 0" $nnotm $work
Check ($o -match 'only valid for N = 2\^k - 1') 'a non-Mersenne N is refused with the fold-specific error'
Check ($o -notmatch 'stage1 returned: 0') 'the non-Mersenne run did NOT run stage 1'
$o = Run-Tool $FoldExe "-gpu -d $Device --gpu-param 3 -sigma 3:5555 -gpucurves 8 -savea $work\E3.save 1e4 0" $n991 $work
# the message spans two source lines, each of which the logger prefixes with a timestamp,
# so only the fragment that cannot be split is matched here
Check ($o -match 'carries fold kernels') 'gpu_param 3 is refused by a fold build'
Check ($o -notmatch 'stage1 returned: 0') 'the param3 run did NOT run stage 1'

if (-not $SkipSlow) {
    Write-Host ""
    Write-Host "=== F. throughput A/B: M4999, tier 5120, $AbCurves curves, B1=1e5 (device $Device) ==="
    # NOTE the shape matters more than anything else in this comparison: the fold is
    # latency-bound, so it only reaches parity/lead when the batch keeps >= 4 blocks per
    # SM resident.  384 curves on the 24-SM 4060 = 2 blocks/SM (fold ~7% behind), 768 =
    # 4 blocks/SM (parity), and with the fold family's own TPB (256) it leads by 3-8%.
    # docs/ECM_CGBN_OPTIMIZATION.md 9.9.
    $fF = Join-Path $work 'F_fold.save'; $mF = Join-Path $work 'F_mont.save'
    $tf = @(); $tm = @()
    foreach ($rep in 1, 2) {
        $o = Run-Tool $FoldExe "-gpu -d $Device --gpu-param 0 -sigma 300000 -gpucurves $AbCurves -savea $fF.rep$rep 1e5 0" $n4999 $work
        $tf += (Get-GpuTime $o)
        $o = Run-Tool $CudaExe "-gpu -d $Device --gpu-param 0 -sigma 300000 -gpucurves $AbCurves -savea $mF.rep$rep 1e5 0" $n4999 $work
        $tm += (Get-GpuTime $o)
    }
    $bestF = ($tf | Measure-Object -Minimum).Minimum
    $bestM = ($tm | Measure-Object -Minimum).Minimum
    Write-Host ("  fold  build gputime: {0} ms" -f ($tf -join ' / '))
    Write-Host ("  mont  build gputime: {0} ms" -f ($tm -join ' / '))
    if ($bestF -gt 0 -and $bestM -gt 0) {
        $spd = [Math]::Round(($bestM / $bestF - 1) * 100, 2)
        Write-Host ("  best-of-2: mont {0:N0} ms vs fold {1:N0} ms -> fold is {2}% faster" -f $bestM, $bestF, $spd)
        # The measured state of the art is that the fold is SLOWER end-to-end (-15..-19%,
        # docs/ECM_CGBN_OPTIMIZATION.md 9.2/9.4): CGBN's mont_mul interleaves its product and
        # reduction chains, while the fold exposes one serial chain.  So a speedup is NOT
        # expected here; -ExpectFoldFaster flips this into a gate for whoever improves it.
        if ($ExpectFoldFaster) {
            Check ($bestF -lt $bestM) "the fold build is faster on M4999 ($spd% by best-of-2)"
        } else {
            Check $true ("throughput recorded (fold {0:N0} ms vs mont {1:N0} ms = {2}%; the fold is " -f $bestF, $bestM, $spd) +
                        "expected to be SLOWER here, see docs 9.4 -- pass -ExpectFoldFaster to gate on it)"
        }
    } else {
        Check $false "could not read gputime from both builds"
    }
    $r = Compare-Saves "$fF.rep1" "$mF.rep1"
    Check (($r.a -eq $AbCurves) -and ($r.same -eq $AbCurves) -and ($r.diff -eq 0)) "the timed M4999 runs agree line for line ($($r.same)/$($r.a) of $AbCurves)"
}

Write-Host ""
if ($fails -eq 0) { Write-Host 'ALL OK' -ForegroundColor Green } else { Write-Host "FAILURES: $fails" -ForegroundColor Red }
exit $fails
