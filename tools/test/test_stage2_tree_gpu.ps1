#Requires -Version 5.1
<#
.SYNOPSIS
    GPU tree-based stage 2 (slice S1/S2): is the GPU F tree the CPU reference's F tree?

.DESCRIPTION
    tools/bench/stage2_tree_gpu.cu builds the baby product tree F(X) = prod_j (X - x_j) mod N
    on the GPU, reusing the SAME NTT multiply kernels as tools/bench/ntt_poly_probe.cu
    (extracted into ntt_poly_mul_host(), so there is exactly one implementation of every
    kernel -- two copies would drift, which is what docs/DEV_STAGE2_GPU_PLAN.md section 14.8
    records as a lesson).  The oracle is coefficient-by-coefficient:

      [1] the device x-only ladder reproduces the CPU reference's baby points x_j
      [2] the GPU F equals the CPU F from `stage2_tree_ref --dump-F`, mod N, coefficient by
          coefficient, for TWO different shapes (D=210 -> P=24, D=2310 -> P=240)
      [3] the exactness bound L*(2^bpw-1)^2 < p is re-derived for the tree's OWN shapes (it
          must not be inherited from the probe's shapes)
      [4] the frozen factor is found with the SAME hit prime as the CPU reference, and
          B2=114000 finds nothing on both sides
      [5] no regressions in the two existing NTT/tree suites

    This wrapper drives the acceptance script the slice shipped with
    (tools/build/check_stage2_tree_gpu.ps1) and asserts on its output, so there is one place
    that defines acceptance and one place that gates it.  Uses device 1 (device 0 normally
    runs production stage 1).  Skips cleanly when the exe or the script is missing.

.PARAMETER Exe
    stage2_tree_gpu.exe.  Default: <repo>\build_cuda_cmake\stage2_tree_gpu.exe

.PARAMETER Device
    CUDA device (default 1: device 0 runs the user's production stage-1 job).
#>
param(
    [string]$Exe = '',
    [string]$Check = '',
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
if (-not $Exe) { $Exe = Join-Path $repoRoot 'build_cuda_cmake\stage2_tree_gpu.exe' }
if (-not $Check) { $Check = Join-Path $repoRoot 'tools\build\check_stage2_tree_gpu.ps1' }

if (-not (Test-Path $Exe)) {
    Write-Host "[skip] stage2_tree_gpu.exe not found -- build it with tools\build\build_stage2_tree_gpu.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "passed: 0   failed: 0"
    exit 0
}
if (-not (Test-Path $Check)) {
    Write-Host "FAIL: the acceptance script is missing ($Check)" -ForegroundColor Red
    exit 2
}

Write-Host "GPU tree stage 2 (S1: F tree vs the CPU reference; S2: frozen factor)"
Write-Host ("exe    : " + $Exe)
Write-Host ("device : " + $Device)
Write-Host ""

# The acceptance script takes -N/-Sigma/-B1/-B2/-D/-Device and a -Evaluate switch that also
# walks the stage-2 tail (giant points, remainder tree, gcd).  It has no -Exe parameter: it
# locates the exe itself, so we only pass shapes and the device.
function RunCheck([string[]]$extra) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $o = (& powershell -NoProfile -ExecutionPolicy Bypass -File $Check -Device $Device @extra 2>&1 | Out-String)
    $c = $LASTEXITCODE
    $ErrorActionPreference = $prev
    return @{ out = $o; code = $c }
}
$r210 = RunCheck @('-D', '210')
$r2310 = RunCheck @('-D', '2310')
$rEval = RunCheck @('-D', '210', '-Evaluate')
$rSharp = RunCheck @('-D', '210', '-Evaluate', '-B2', '114000')
# [6] the naming cap must NOT lose a factor (docs/DEV_STAGE2_GPU_PLAN.md sections 26.3/26.5):
# with NTT_NAME_MAX=1 the candidate scan stops after the first hit leaf, but every OTHER hit
# leaf's factor is recorded from its own gcd.  Before that fix the reported factor set depended
# on the diagnostic scan, so a capped run could report NO factor at all -- and at B2=1e11 a full
# scan is ~26 hours, i.e. the cap is not optional in production.
$rCap = RunCheck @('-D', '210', '-Evaluate', '-NameMax', '1')
# [7] the giant-point DIFFERENTIAL-ADDITION CHAIN (objective 3, section 31) is now the default for
# large chunks, so it gets its own acceptance: forced on the frozen vector together with its own
# point-by-point comparison against the per-point ladder.
$rChain = RunCheck @('-D', '210', '-Evaluate', '-ChainCheck')
# [8] the two OPERAND PACKERS must agree end to end (section 33): the device packer is the
# default and the host packer is the oracle, and the frozen shape is where both can be run
# against the CPU reference's factor set.
$rHostPack = RunCheck @('-D', '210', '-Evaluate', '-HostPack')
$out = $r210.out + "`n" + $r2310.out + "`n" + $rEval.out + "`n" + $rSharp.out
$code = $rEval.code

# [1] ladder: the device's x_j must equal the CPU reference's
$m = [regex]::Match($out, 'ladder: baby_points=(\d+) mismatches=(\d+)')
Check "the device ladder reports its baby points and mismatches" ($m.Success) $out.Trim()
if ($m.Success) {
    Check "every device baby point x_j equals the CPU reference's" ($m.Groups[2].Value -eq '0') $m.Value
    Check "the baby set is non-empty (>= 24 points)" ([int]$m.Groups[1].Value -ge 24) $m.Value
}

# [2] F coefficient by coefficient, for every shape the script walks
$fchecks = [regex]::Matches($out, 'check_F: coeffs_gpu=(\d+) coeffs_cpu=(\d+) mismatches=(\d+)')
Check "check_F ran (at least two shapes)" ($fchecks.Count -ge 2) ("check_F lines=" + $fchecks.Count)
$allF = $true
foreach ($c in $fchecks) {
    if ($c.Groups[3].Value -ne '0') { $allF = $false }
    if ($c.Groups[1].Value -ne $c.Groups[2].Value) { $allF = $false }
}
Check "every GPU F coefficient equals the CPU F's, mod N, on every shape" $allF `
      (($fchecks | ForEach-Object { $_.Value }) -join ' ; ')
Check "the two F dumps contain no differing lines" ($out -match 'differing lines: 0' -or $out -match 'ok=1')

# [3] the exactness bound, re-derived for the tree's own shapes
Check "the exactness bound is re-derived for the tree's own shapes" `
      ($out -match 'exactness: binding_shape P=\d+ S=\d+.*L=P\*slot_words=\d+ bpw=\d+.*< p') `
      $out.Trim()

# [4] the frozen vector: same factor and same hit prime as the CPU reference, plus sharpness
Check "the frozen 17-digit factor is found" ($out -match 'factors=59649589127497217') $out.Trim()
Check "the hit prime is the same one the CPU reference names (114713)" ($out -match 'hit_primes=114713') $out.Trim()
Check "no reported factor fails to divide N" ($out -match 'bad_factors=0') $out.Trim()
Check "B2=114000 finds nothing (sharpness, both sides)" ($out -match 'hits=0') $out.Trim()

# [5] the acceptance script's own verdict and exit code
Check "the acceptance script reports all checks passed" ($out -match 'all checks passed') $out.Trim()
Check "the acceptance script exits 0" ($code -eq 0) ("exit=" + $code)

# [6] the capped-naming regression: the factor set must survive a bounded diagnostic scan
Check "the capped run (NTT_NAME_MAX=1) still finds the frozen factor" `
      ($rCap.out -match 'factors=59649589127497217') $rCap.out.Trim()
Check "the capped run reports no bogus factor" ($rCap.out -match 'bad_factors=0') $rCap.out.Trim()
Check "the capped run's acceptance script exits 0" ($rCap.code -eq 0) ("exit=" + $rCap.code)
Check "the naming accounting line is printed (hit_leaves / unnamed / t_scan / t_ladder)" `
      ($rCap.out -match 'batched_naming: hit_blocks=\d+ hit_leaves=\d+ named_searches=\d+ candidates_tested=\d+ unnamed=\d+ t_scan=[\d.]+ t_ladder=[\d.]+ t_name=[\d.]+ name_max=\d+') `
      $rCap.out.Trim()

# [7] the giant-point chain (objective 3): exact on every point, and really used
Check "the giant chain reproduces the ladder point by point (mismatches=0)" `
      ($rChain.out -match 'giant_chain_check: points=\d+ .*mismatches=0') $rChain.out.Trim()
Check "the giant chain was really used (chunks >= 1)" `
      ($rChain.out -match '(real|batched)_giant_chain: chunks=[1-9]') $rChain.out.Trim()
Check "the chained run still finds the frozen factor" `
      ($rChain.out -match 'factors=59649589127497217') $rChain.out.Trim()

# [8] the device packer (default) and the host packer (oracle) must agree on the factor set
Check "the host-packing oracle still finds the frozen factor (device packer's A/B)" `
      ($rHostPack.out -match 'factors=59649589127497217' -and `
       $rHostPack.out -match 'hit_primes=114713') $rHostPack.out.Trim()
Check "the host-packing A/B exits 0" ($rHostPack.code -eq 0) ("exit=" + $rHostPack.code)

# [9] THE S5 DEVICE DESCENT (objective 2, docs/DEV_STAGE2_GPU_PLAN.md sections 49-54): the walk
# must be aligned with the host's AND every leaf must match it, and only then may NTT_S5_ON be
# trusted.  No carry-round override is needed any more: the shape planner used to round the bpw
# DOWN to a divisor of the (often prime) slot width and could land on bpw = 1, whose single-bit
# digits make the carry's chain O(N) -- 128 rounds were needed to work around it.  Rounding the
# SLOT UP instead removes that degradation entirely, so this group now runs the DEFAULT shape.
$env:NTT_S5_ON = '1'
$env:NTT_S4_DESCENT_CHECK = '1'
$rS5 = RunCheck @('-D', '210', '-Evaluate')
$rS5big = RunCheck @('-D', '2310', '-B1', '1000', '-B2', '400000', '-Evaluate')
# ... and THE SAME LARGER SHAPE with S5 OFF (the descent CHECK stays on, because these lines are
# the assertion), which is a regression guard for the leaf-count fix (section 54.5): the descent's
# leaf count used to come from `H.size()`, and at this shape the whole giant set is ONE block of
# 175 giant points while P = 240, so H is legitimately SHORT (`H = T`, no reduction) and every
# consumer that walked P rows walked past the descent's output.  The host-descent path must
# therefore be exact here -- walking 64 rows past `ref` was a host access violation before.
Remove-Item Env:\NTT_S5_ON -ErrorAction SilentlyContinue
$rBigNoS5 = RunCheck @('-D', '2310', '-B1', '1000', '-B2', '400000', '-Evaluate')
Remove-Item Env:\NTT_S4_DESCENT_CHECK -ErrorAction SilentlyContinue
Check "S5 (device descent): every leaf equals the host descent's" `
      ($rS5.out -match 'descent_check_leaves: P=\d+ differing_leaves=0') $rS5.out.Trim()
Check "S5: every coefficient equals the host descent's" `
      ($rS5.out -match 'descent_check: P=\d+ divmods_batched=\d+ divmods_slow=\d+ mismatching_coefficients=0') `
      $rS5.out.Trim()
Check "S5: the walk is aligned with the host's op count" `
      ($rS5.out -match 'descent_check: .*divmods_batched=(\d+) divmods_slow=(\d+)') $rS5.out.Trim()
# The device reports its ops in four classes; the host's `divmods` counts every child that is
# neither the padding subtree nor a degree fast path, i.e. generic + linear.  Asserting THAT
# identity is the alignment claim (section 45.2) -- asserting `divmods_batched == divmods_slow`
# compares the generic count against the total and is simply the wrong test.
$mSt = [regex]::Match($rS5.out, 'descent_dev_stats: divmods=\d+ generic=(\d+) linear=(\d+)')
$mSl = [regex]::Match($rS5.out, 'descent_check: .*divmods_slow=(\d+)')
if ($mSt.Success -and $mSl.Success) {
    $gen = [int]$mSt.Groups[1].Value
    $lin = [int]$mSt.Groups[2].Value
    $slow = [int]$mSl.Groups[1].Value
    Check "S5: generic+linear equals the host's op count ($gen+$lin = $slow)" `
          (($gen + $lin) -eq $slow) ("generic=$gen linear=$lin slow=$slow")
}
Check "S5: the acceptance script exits 0" ($rS5.code -eq 0) ("exit=" + $rS5.code)
# THE LEAF-COUNT FIX (section 54.5), asserted where it was found: at D=2310 with S5 OFF the host
# descent must be exact AND must still find the frozen factor.  Both were violated by `P =
# H.size()`; the descent check CRASHED (host 0xC0000005) on the 64 rows the descent never produced.
Check "D=2310 (S5 off): the host descent is exact (leaf-count regression guard)" `
      ($rBigNoS5.out -match 'descent_check_leaves: P=240 differing_leaves=0' -and `
       $rBigNoS5.out -match 'descent_check: P=240 divmods_batched=478 divmods_slow=478 mismatching_coefficients=0') `
      $rBigNoS5.out.Trim()
Check "D=2310 (S5 off): the frozen factor is still found" `
      ($rBigNoS5.out -match 'factors=59649589127497217' -and $rBigNoS5.out -match 'bad_factors=0') `
      $rBigNoS5.out.Trim()
Check "D=2310 (S5 off): the acceptance script exits 0" ($rBigNoS5.code -eq 0) ("exit=" + $rBigNoS5.code)
# ---- S5 ON A SHAPE 10x THE FROZEN VECTOR (section 55) ---------------------------------------
# These two checks were the INVERTED canary of section 54.6 ("S5 at D=2310 is still a known
# defect") until round 21 found and fixed the cause: `s5_sub_kernel` guards on `rows*nw` and
# derives its row as `gid/nw`, but the call site launched S5_GRID(rows) -- ceil(rows/256)*256
# threads instead of rows*nw.  At D=2310 the root division has rows = 128 and nw = 3, so only
# gid <= 255 existed and rows 0..85 were written while rows 86..127 kept their previous contents
# (86 = floor(255/3), exactly the first coefficient the descent check flagged).  Every smaller
# shape fits in one block (D=210: rows = 16 -> 48 threads; D=2310 level 7: rows = 64 -> 192), which
# is why a P=24 vector could not see it.  Now that it is fixed these are positive assertions, and
# they are the reason this group carries a second, larger shape at all.
Check "S5 on D=2310: every leaf equals the host descent's" `
      ($rS5big.out -match 'descent_check_leaves: P=240 differing_leaves=0') $rS5big.out.Trim()
Check "S5 on D=2310: every coefficient equals the host descent's" `
      ($rS5big.out -match 'descent_check: P=240 divmods_batched=238 divmods_slow=478 mismatching_coefficients=0') `
      $rS5big.out.Trim()
Check "S5 on D=2310: the batched engine finds the frozen factor" `
      ($rS5big.out -match 'algorithm=tree_gpu_batched .*hits=1 bad_factors=0 factors=59649589127497217 hit_primes=114713') `
      $rS5big.out.Trim()
Check "S5 on D=2310: the acceptance script exits 0" ($rS5big.code -eq 0) ("exit=" + $rS5big.code)

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
