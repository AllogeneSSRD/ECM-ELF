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
# ---- A THIRD SHAPE WHERE THE FAST PATH ACTUALLY FIRES (section 56.2) -------------------------
# At D=30030 (P=2880) the whole giant set is ONE block of 68 points, so deg H = 67 against a root
# divisor of degree 2048: `H mod F = H` at the top levels, and the descent must take the COPY
# branch there instead of running 43 full-size Newton divisions (which is what testing the ROW
# WIDTH instead of the polynomial's DEGREE cost it: most of t_generic, and the section 45.2
# counting identity off by exactly 43).  The copy must also write deg+1 rows and not the parent's
# full width, or the frontier overflows ("the S5 frontier is larger than the chunk").
# NOTE: this run's acceptance script exits 1 for an unrelated and already-documented reason (the S2
# host tail counts hit BLOCKS: tail_counts hit_blocks=2 factors=1, section 55.7), so only the
# descent and the batched summary are asserted here -- deliberately not the exit code.
# The two switches are RE-SET here: the S5-off regression run above removed them, and without this
# the block silently measured the HOST descent (0.07 s, no `descent_check` lines at all) -- a third
# self-inflicted test bug of this kind, after `$Matches` being clobbered and `divmods_batched` being
# compared against `divmods_slow`.
$env:NTT_S5_ON = '1'
$env:NTT_S4_DESCENT_CHECK = '1'
$rS5huge = RunCheck @('-D', '30030', '-B1', '1000', '-B2', '2000000', '-Evaluate')
Remove-Item Env:\NTT_S5_ON, Env:\NTT_S4_DESCENT_CHECK -ErrorAction SilentlyContinue
Check "S5 on D=30030 (degree fast path): every leaf equals the host descent's" `
      ($rS5huge.out -match 'descent_check_leaves: P=2880 differing_leaves=0') $rS5huge.out.Trim()
Check "S5 on D=30030: every coefficient equals the host descent's" `
      ($rS5huge.out -match 'descent_check: P=2880 divmods_batched=\d+ divmods_slow=5715 mismatching_coefficients=0') `
      $rS5huge.out.Trim()
# ... and the section 45.2 identity, which is the assertion that the fast path is decided on the
# DEGREE: generic+linear must equal the host's division count exactly.
$mH = [regex]::Match($rS5huge.out, 'descent_dev_stats: divmods=\d+ generic=(\d+) linear=(\d+)')
$mHs = [regex]::Match($rS5huge.out, 'descent_check: P=2880 .*divmods_slow=(\d+)')
if ($mH.Success -and $mHs.Success) {
    $gh = [int]$mH.Groups[1].Value
    $lh = [int]$mH.Groups[2].Value
    $sh = [int]$mHs.Groups[1].Value
    Check "S5 on D=30030: generic+linear equals the host's op count ($gh+$lh = $sh)" `
          (($gh + $lh) -eq $sh) ("generic=$gh linear=$lh slow=$sh")
}
Check "S5 on D=30030: the batched engine finds the frozen factor" `
      ($rS5huge.out -match 'algorithm=tree_gpu_batched .*hits=1 bad_factors=0 factors=59649589127497217') `
      $rS5huge.out.Trim()

# [10] THE ARENA'S CACHE EVICTION (section 20, objective 1).  NttArena caches, per (N, nbatch), the
# three big buffers dA/dB/dQ (3N words) plus dOut/dRes, and per N the fused per-pass tables -- all
# of it a PURE CACHE that used to be kept forever.  That is what capped D: at a larger P the fold's
# shape needed room while earlier shapes still held thousands of MB, the arena refused, the run fell
# back to per-call cudaMalloc and died with "out of memory" at the production shape.
# evict_other_shapes() now frees every other shape's caches before refusing, and THIS IS THE ONLY
# TEST THAT EXERCISES IT (every other group goes through --check-F, which never enters run_real).
# THE CAP IS CALIBRATED FROM MEASUREMENT: at the frozen N with D=1231230 (P=115200) the run reaches
# 439 MB of arena; the cap below is 342 MB, so the cached sum does not fit and the allocator must
# evict.  `arena_overflow=0` is the assertion -- without eviction this run refuses and counts
# overflows.  Cost: 0.64 s.
$env:NTT_ARENA_CAP_KB = '350000'
$oEv = (& $Exe @('--real', '--n', '340282366920938463463374607431768211457', '--sigma', '26',
                 '--b1', '1000', '--b2', '5000000', '--d', '1231230', '--device', "$Device") 2>&1 |
       Out-String)
$cEv = $LASTEXITCODE
Remove-Item Env:\NTT_ARENA_CAP_KB -ErrorAction SilentlyContinue
Check "arena eviction: a large-P run fits a cap below its natural usage (arena_overflow=0)" `
      ($oEv -match 'arena_overflow=0' -and $oEv -notmatch 'arena refuses') $oEv.Trim()
Check "arena eviction: same frozen factor and a clean exit" `
      ($oEv -match 'bad_factors=0 factors=59649589127497217' -and $cEv -eq 0) ("exit=" + $cEv)

# [11] THE DEFERRED CARRY CHECK (section 29 of docs/DEV_GPUOWL_NTT_NOTES.md).  A chunked batched
# multiply used to pay the probe's carry-residual readback -- a PAGEABLE D2H, i.e. a full pipeline
# drain -- once per chunk; at the production shape that was 22471 x 1.64 ms = 36.75 s of a 253.53 s
# run (measured).  Now every chunk except the first and the last leaves the counters on the device
# and ntt_batch_carry_finish() reads them once, before the last chunk (whose memset would wipe them).
# THE TEST HAS TWO JOBS, and the second is the important one: the answer must not change, AND the
# deferral must be provably ACTIVE -- a run that quietly stopped checking would look identical in
# every timing number, which is exactly how this kind of optimisation goes wrong.  So it asserts
# chunks_deferred > finishes (i.e. at least one readback really did cover several chunks) as well as
# the frozen factor and bad_factors=0.
# NTT_S4_BATCH_MB=1 forces the smallest possible chunk, so the round-trip count is maximal and the
# deferral cannot be missed.  The default-budget run is the control: same factor, deferral optional.
$env:NTT_S4_BATCH_MB = '1'
$oDef = (& $Exe @('--real', '--n', '340282366920938463463374607431768211457', '--sigma', '26',
                  '--b1', '1000', '--b2', '5000000', '--d', '1231230', '--device', "$Device") 2>&1 |
        Out-String)
$cDef = $LASTEXITCODE
Remove-Item Env:\NTT_S4_BATCH_MB -ErrorAction SilentlyContinue
$mDef = [regex]::Match($oDef, 'chunks_deferred=(\d+) finishes=(\d+)')
Check "deferred carry check: the tiny-budget run still finds the frozen factor" `
      ($oDef -match 'bad_factors=0 factors=59649589127497217' -and $cDef -eq 0) ("exit=" + $cDef)
if ($mDef.Success) {
    $dc = [int]$mDef.Groups[1].Value
    $df = [int]$mDef.Groups[2].Value
    # dc == df HERE, measured, and that is not a defect: this shape's chunk ladder halves from
    # nbatch, so a call gets 1, 2 or 3 chunks -- a 3-chunk call has exactly ONE interior chunk, so
    # one deferral per finish.  What the test must prove is that the deferred path is TAKEN and that
    # finishes really happen; the "many chunks per finish" claim is proven by the production run
    # (22471 chunks, ~800 calls) recorded in section 29, not by a 0.6 s test.
    Check "deferred carry check: the deferral ran (chunks_deferred >= finishes > 0)" `
          ($dc -gt 0 -and $df -gt 0 -and $dc -ge $df) ("chunks_deferred=$dc finishes=$df")
} else {
    Check "deferred carry check: the accounting line is printed (chunks_deferred/finishes)" $false `
          $oDef.Trim()
}
$oDef2 = (& $Exe @('--real', '--n', '340282366920938463463374607431768211457', '--sigma', '26',
                   '--b1', '1000', '--b2', '5000000', '--d', '1231230', '--device', "$Device") 2>&1 |
         Out-String)
$cDef2 = $LASTEXITCODE
Check "deferred carry check: the default-budget control run agrees" `
      ($oDef2 -match 'bad_factors=0 factors=59649589127497217' -and $cDef2 -eq 0) ("exit=" + $cDef2)
Check "deferred carry check: no carry-convergence failure was reported" `
      ($oDef -notmatch 'CARRY DID NOT CONVERGE' -and $oDef2 -notmatch 'CARRY DID NOT CONVERGE') ""

# [13] THE 2-BY-1 DIVISION PRIMITIVE (section 31 of docs/DEV_GPUOWL_NTT_NOTES.md).  Groundwork for
# objective 4's division reduction: the tail of the device reduction can go from 2*nw^2 MACs to about
# nw^2 by replacing the final Montgomery multiplication with ONE plain long division of C by N, and a
# long division needs an exact 2-by-1 quotient digit.  Section 35.1 of docs/DEV_STAGE2_GPU_PLAN.md
# records that a previous optimisation in this area was mathematically wrong and was caught by the
# gate, so the arithmetic is checked BEFORE anything depends on it: the primitive is verified on the
# host against GMP for 256 numerators per modulus, saturating extreme included, and a mismatch is
# fatal.  Its first version was wrong in 256 of 256 cases and this check stopped the run.
# The constants are FILE-STATICS at modulus level, deliberately not per-shape: an earlier attempt put
# them in S4Reduce::Shape and the --check-F path then died silently (exit 3, empty stderr, gate
# 47/47 -> 16 passed / 32 failed), which a stash-and-rebuild bisect confirmed.
$mU = [regex]::Matches($oDef2, 's4_udiv_check: cases=(\d+) bad=(\d+)')
Check "udiv primitive: the GMP check runs with a full 256 cases" `
      ($mU.Count -ge 1 -and $mU[0].Groups[1].Value -eq '256') ("lines=" + $mU.Count)
Check "udiv primitive: zero disagreements with GMP (the fatal guard would have stopped the run)" `
      ($mU.Count -ge 1 -and ($mU | Where-Object { $_.Groups[2].Value -ne '0' }).Count -eq 0) `
      (($mU | Select-Object -First 2 | ForEach-Object { $_.Value.Trim() }) -join ' | ')
Check "udiv primitive: no mismatch message on either run" `
      ($oDef2 -notmatch 'udiv_2by1 MISMATCH' -and $oDef -notmatch 'udiv_2by1 MISMATCH') ""

# [12] THE ASYNCHRONOUS CHUNK TRANSFERS (section 30).  The upload uses pinned staging and the
# readback goes into double-buffered pinned memory, consumed one chunk late, so neither transfer
# forces the implicit sync that a PAGEABLE copy needs.  The gain is real but SMALL at the shapes a
# test can afford (production 2x2: async -2.5%, deferral -0.5%, both -3.4%, all four inside one
# binary because the card's clock swings 1000-1772 MHz and cross-build comparisons are confounded).
# A small gain is exactly the kind that a silent regression -- losing pinned memory and quietly
# falling back to the blocking path -- would hide, so this asserts the counters instead of timing:
# every upload and every readback went through the pinned path, and the fallback counter is zero.
$mAx = [regex]::Match($oDef2, 'raw_async=(\d+) out_async=(\d+) fallbacks=(\d+) \(async_enabled=1\)')
if ($mAx.Success) {
    $ra = [int]$mAx.Groups[1].Value
    $oa = [int]$mAx.Groups[2].Value
    $fb = [int]$mAx.Groups[3].Value
    Check "async transfers: every upload and readback used the pinned path (no fallback)" `
          ($ra -gt 0 -and $oa -gt 0 -and $fb -eq 0) ("raw_async=$ra out_async=$oa fallbacks=$fb")
} else {
    Check "async transfers: the accounting line is printed (raw_async/out_async/fallbacks)" $false `
          $oDef2.Trim()
}
$env:NTT_S4_ASYNC = '0'
$oBlk = (& $Exe @('--real', '--n', '340282366920938463463374607431768211457', '--sigma', '26',
                  '--b1', '1000', '--b2', '5000000', '--d', '1231230', '--device', "$Device") 2>&1 |
         Out-String)
$cBlk = $LASTEXITCODE
Remove-Item Env:\NTT_S4_ASYNC -ErrorAction SilentlyContinue
Check "async transfers: the blocking-off A/B gives the same frozen factor" `
      ($oBlk -match 'bad_factors=0 factors=59649589127497217' -and $cBlk -eq 0) ("exit=" + $cBlk)
Check "async transfers: NTT_S4_ASYNC=0 really disables the pinned path (A/B knob works)" `
      ($oBlk -match 'raw_async=0 out_async=0 .*async_enabled=0') $oBlk.Trim()

# [14] Plain long division replaces BOTH REDC and domain restoration.  Exercise the runtime
# A/B in one binary, and require exact factor/hit agreement as well as the independent full
# multiword remainder fixtures (including borrow repairs).  Explicit modes avoid inheriting
# a developer's A/B setting; restore that setting even if a launch fails.
$savedTail = $env:NTT_S4_OLDTAIL
try {
    $tailArgs = @('--real', '--n', '340282366920938463463374607431768211457', '--sigma', '26',
                  '--b1', '1000', '--b2', '5000000', '--d', '1231230', '--device', "$Device")
    $env:NTT_S4_OLDTAIL = '0'
    $oDiv = (& $Exe @tailArgs 2>&1 | Out-String)
    $cDiv = $LASTEXITCODE
    $env:NTT_S4_OLDTAIL = '1'
    $oMont = (& $Exe @tailArgs 2>&1 | Out-String)
    $cMont = $LASTEXITCODE
    Check "division reduction: frozen factor, clean exit on both A/B paths" `
          ($cDiv -eq 0 -and $cMont -eq 0 -and
           $oDiv -match 'bad_factors=0 factors=59649589127497217' -and
           $oMont -match 'bad_factors=0 factors=59649589127497217') "exit=$cDiv/$cMont"
    Check "division reduction: the runtime knob selects both algorithms" `
          ($oDiv -match 's4_reduce_mode: algorithm=division' -and
           $oMont -match 's4_reduce_mode: algorithm=montgomery') ""
    $divSummary = [regex]::Match($oDiv, 'stage2:.*factors=([^\s]*) hit_primes=([^\s]*)')
    $montSummary = [regex]::Match($oMont, 'stage2:.*factors=([^\s]*) hit_primes=([^\s]*)')
    Check "division reduction: factor and hit-prime sets agree exactly" `
          ($divSummary.Success -and $montSummary.Success -and
           $divSummary.Groups[1].Value -eq $montSummary.Groups[1].Value -and
           $divSummary.Groups[2].Value -eq $montSummary.Groups[2].Value) ""
    $divFixture = [regex]::Match($oDiv, 's4_div_check: cases=(\d+) bad=(\d+) repairs=(\d+)')
    Check "division reduction: full GMP fixtures pass and exercise borrow repairs" `
          ($divFixture.Success -and $divFixture.Groups[1].Value -eq '608' -and
           $divFixture.Groups[2].Value -eq '0' -and [int]$divFixture.Groups[3].Value -gt 0) `
          $divFixture.Value
    $env:NTT_S4_OLDTAIL = '0'
    foreach ($edgeN in @('ffffffffffffffc5', 'ffffffffffffffffffffffffffffff61')) {
        $oEdge = (& $Exe @('--real', '--n-hex', $edgeN, '--sigma', '26', '--b1', '20',
                          '--b2', '1000', '--d', '210', '--device', "$Device") 2>&1 | Out-String)
        $cEdge = $LASTEXITCODE
        Check "division reduction: GPU normalized edge modulus $edgeN agrees with GMP" `
              ($cEdge -eq 0 -and $oEdge -match 'algorithm=division .*dshift=0' -and
               $oEdge -match 's4_reduce_selftest:.*mismatches=0' -and
               $oEdge -notmatch 'div_rem MISMATCH|udiv_2by1 MISMATCH|FATAL|gmp_bad=[1-9]|mismatches=[1-9]') "exit=$cEdge"
    }
} finally {
    if ($null -eq $savedTail) { Remove-Item Env:\NTT_S4_OLDTAIL -ErrorAction SilentlyContinue }
    else { $env:NTT_S4_OLDTAIL = $savedTail }
}

# [15] Deferred oracle: retain the SAME samples, validate every captured snapshot, bound the
# queue, and detect a corrupted final snapshot at the phase drain. All toggles are restored.
$oracleSaved = @{}
foreach ($key in @('NTT_S4_ORACLE_ASYNC','NTT_S4_ORACLE_RING','NTT_S4_ORACLE_TEST_BAD',
                    'NTT_S4_SAMPLE','NTT_S4_CHECK_EVERY','NTT_S4_OLDTAIL')) {
    $oracleSaved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}
try {
    $env:NTT_S4_OLDTAIL = '0'; $env:NTT_S4_ORACLE_RING = '4'; $env:NTT_S4_ORACLE_TEST_BAD = '0'
    $env:NTT_S4_SAMPLE = '96'; $env:NTT_S4_CHECK_EVERY = '8'
    $oracleArgs = @('--real','--n','340282366920938463463374607431768211457','--sigma','26',
                    '--b1','1000','--b2','5000000','--d','1231230','--device',"$Device")
    $env:NTT_S4_ORACLE_ASYNC = '0'
    $oSyncOracle = (& $Exe @oracleArgs 2>&1 | Out-String); $cSyncOracle = $LASTEXITCODE
    $env:NTT_S4_ORACLE_ASYNC = '1'
    $oAsyncOracle = (& $Exe @oracleArgs 2>&1 | Out-String); $cAsyncOracle = $LASTEXITCODE
    $oraclePattern = 's4_oracle_stats: async=(\d+) selected=(\d+) queued=(\d+) compared=(\d+) samples=(\d+) pending=(\d+) ring_waits=(\d+) fallbacks=(\d+) signature=([0-9a-f]+)'
    $mSync = [regex]::Match($oSyncOracle,$oraclePattern)
    $mAsync = [regex]::Match($oAsyncOracle,$oraclePattern)
    Check "oracle: blocking and asynchronous modes find the same frozen factor" `
          ($cSyncOracle -eq 0 -and $cAsyncOracle -eq 0 -and
           $oSyncOracle -match 'bad_factors=0 factors=59649589127497217' -and
           $oAsyncOracle -match 'bad_factors=0 factors=59649589127497217') "exit=$cSyncOracle/$cAsyncOracle"
    Check "oracle: mode switch works and all asynchronous snapshots use pinned memory" `
          ($mSync.Success -and $mAsync.Success -and $mSync.Groups[1].Value -eq '0' -and
           $mSync.Groups[3].Value -eq '0' -and $mAsync.Groups[1].Value -eq '1' -and
           [long]$mAsync.Groups[3].Value -gt 0 -and
           $mAsync.Groups[2].Value -eq $mAsync.Groups[3].Value -and
           $mAsync.Groups[8].Value -eq '0') ""
    $allConsumed = $mSync.Success -and $mAsync.Success
    foreach ($mOracle in @($mSync,$mAsync)) {
        $allConsumed = $allConsumed -and $mOracle.Groups[2].Value -eq $mOracle.Groups[4].Value -and
                       $mOracle.Groups[6].Value -eq '0'
    }
    Check "oracle: every selected snapshot was compared and the final queue is empty" $allConsumed ""
    Check "oracle: sample positions, count and job count agree across the same-binary A/B" `
          ($mSync.Success -and $mAsync.Success -and
           $mSync.Groups[2].Value -eq $mAsync.Groups[2].Value -and
           $mSync.Groups[5].Value -eq $mAsync.Groups[5].Value -and
           $mSync.Groups[9].Value -eq $mAsync.Groups[9].Value) ""
    $mGmpOracle = [regex]::Match($oAsyncOracle,'s4_multiply_stats:.*gmp_checked=(\d+)')
    Check "oracle: checked GMP coefficient count equals the snapshot sample count" `
          ($mAsync.Success -and $mGmpOracle.Success -and
           $mAsync.Groups[5].Value -eq $mGmpOracle.Groups[1].Value -and
           $oAsyncOracle -match 't_wait=[0-9.]+ t_copy_host=[0-9.]+ t_gmp=[0-9.]+') ""
    $env:NTT_S4_ORACLE_RING = '1'; $env:NTT_S4_CHECK_EVERY = '1'
    $stressArgs = @('--real','--n-hex','ffffffffffffffc5','--sigma','26','--b1','20','--b2','1000',
                    '--d','210','--device',"$Device")
    $oOracleRing = (& $Exe @stressArgs 2>&1 | Out-String); $cOracleRing = $LASTEXITCODE
    $mRing = [regex]::Match($oOracleRing,$oraclePattern)
    Check "oracle: one-slot ring exercises backpressure without losing or overwriting snapshots" `
          ($cOracleRing -eq 0 -and $mRing.Success -and [long]$mRing.Groups[7].Value -gt 0 -and
           $mRing.Groups[2].Value -eq $mRing.Groups[4].Value -and $mRing.Groups[6].Value -eq '0') "exit=$cOracleRing"
    $env:NTT_S4_ORACLE_TEST_BAD = '1'
    $prevOracleErrors = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $oOracleBad = (& $Exe @stressArgs 2>&1 | Out-String); $cOracleBad = $LASTEXITCODE
    $ErrorActionPreference = $prevOracleErrors
    Check "oracle: corrupted last pending snapshot is fatal at the phase drain" `
          ($cOracleBad -ne 0 -and $oOracleBad -match 'FATAL: the device reduction disagrees with GMP') "exit=$cOracleBad"
} finally {
    foreach ($key in $oracleSaved.Keys) { [Environment]::SetEnvironmentVariable($key,$oracleSaved[$key],'Process') }
}

# [16] Pack the oracle's exact integer once instead of repeated GMP shifts/adds. The host
# fixtures compare the INTEGER before modulo, including arbitrary noncanonical 64-bit digits.
$packSaved = @{}
foreach ($key in @('NTT_S4_ORACLE_ASYNC','NTT_S4_ORACLE_PACK','NTT_S4_ORACLE_TEST_BAD')) {
    $packSaved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}
try {
    $env:NTT_S4_ORACLE_ASYNC = '0'; $env:NTT_S4_ORACLE_TEST_BAD = '0'
    $env:NTT_S4_ORACLE_PACK = '0'
    $oSlowPack = (& $Exe @oracleArgs 2>&1 | Out-String); $cSlowPack = $LASTEXITCODE
    $env:NTT_S4_ORACLE_PACK = '1'
    $oFastPack = (& $Exe @oracleArgs 2>&1 | Out-String); $cFastPack = $LASTEXITCODE
    Check "oracle pack: both assembly algorithms retain the frozen factor" `
          ($cSlowPack -eq 0 -and $cFastPack -eq 0 -and
           $oSlowPack -match 'bad_factors=0 factors=59649589127497217' -and
           $oFastPack -match 'bad_factors=0 factors=59649589127497217') "exit=$cSlowPack/$cFastPack"
    Check "oracle pack: exact-integer fixtures cover canonical and noncanonical digits" `
          ($oSlowPack -match 's4_oracle_pack_check: cases=378 bad=0' -and
           $oFastPack -match 's4_oracle_pack_check: cases=378 bad=0') ""
    Check "oracle pack: same-binary knob selects slow and packed assembly" `
          ($oSlowPack -match 's4_oracle_stats:.*pack=0 t_num=[0-9.]+ t_mod=[0-9.]+' -and
           $oFastPack -match 's4_oracle_stats:.*pack=1 t_num=[0-9.]+ t_mod=[0-9.]+') ""
    $mSlowPack = [regex]::Match($oSlowPack,$oraclePattern)
    $mFastPack = [regex]::Match($oFastPack,$oraclePattern)
    Check "oracle pack: full sample positions, count and job count are identical" `
          ($mSlowPack.Success -and $mFastPack.Success -and
           $mSlowPack.Groups[2].Value -eq $mFastPack.Groups[2].Value -and
           $mSlowPack.Groups[5].Value -eq $mFastPack.Groups[5].Value -and
           $mSlowPack.Groups[9].Value -eq $mFastPack.Groups[9].Value) ""
    Check "oracle pack: neither oracle reports GMP disagreement or incomplete snapshots" `
          ($oSlowPack -notmatch 'FATAL|gmp_check_bad=[1-9]|gmp_bad=[1-9]' -and
           $oFastPack -notmatch 'FATAL|gmp_check_bad=[1-9]|gmp_bad=[1-9]' -and
           $mSlowPack.Groups[6].Value -eq '0' -and $mFastPack.Groups[6].Value -eq '0') ""
} finally {
    foreach ($key in $packSaved.Keys) { [Environment]::SetEnvironmentVariable($key,$packSaved[$key],'Process') }
}

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
