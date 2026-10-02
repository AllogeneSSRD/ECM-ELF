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

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
