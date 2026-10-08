#Requires -Version 5.1
<#
.SYNOPSIS
    stage2_tree_ref acceptance -- the TREE (polynomial / Bernstein) ECM stage 2 reference.

.DESCRIPTION
    The tree reference is the ORACLE for the stage-2 structure our GPU engine (Route B,
    docs/DEV_STAGE2_GPU_PLAN.md section 2.2) will implement with an NTT: product tree over
    the baby points, product tree over the giant points, multipoint evaluation by a
    remainder tree.  Layers checked here:

      [1] the tool's own --selftest: the polynomial core (schoolbook multiplication,
          reversed-Newton division with a MONIC divisor, product tree, remainder tree)
          against independent oracles -- pointwise evaluation, q*b + r reconstruction, and
          direct Horner evaluation of F at every point -- over a prime modulus AND over a
          COMPOSITE modulus p1*p2 (the case the algorithm must survive without knowing the
          factors), plus the frozen end-to-end case.
      [2] the frozen vector: N = 2^128+1, sigma=26, B1=1e3, B2=1e6, D=210 must find the
          17-digit factor 59649589127497217, name the SAME stage-2 prime as the pairing
          reference (114713), and produce a factor set EQUAL to
          stage2_ref.exe --algorithm pairing.
      [3] --naive-check on two different (B1,B2,D) shapes: the remainder tree must agree
          with direct Horner evaluation at every giant point.
      [4] sharpness: B2 = 114000 must find NOTHING (the largest prime in the group order is
          114713), and the pairing reference must agree.
      [5] a real --save driven run: the driver's stage-1 save must produce the same factor
          as the ladder path.
      [6] cost accounting: operand_bits is reported with its four-way breakdown, the
          product-tree MODEL is checked against the MEASURED tree cost, and the batched
          (Prime95-shaped) structure is accounted too.
      [7] soundness: every factor that is ever reported divides N (bad_factors=0), on every
          shape including a large one.

    Everything is CPU/GMP, so no GPU is needed.  Deterministic: fixed N, sigma, bounds, D.

.PARAMETER Exe
    stage2_tree_ref.exe.  Default: <repo>\build_cuda_cmake\stage2_tree_ref.exe
.PARAMETER Ref
    stage2_ref.exe (the pairing reference, used for the factor-set equality check).
    Default: <repo>\build_cuda_cmake\stage2_ref.exe
.PARAMETER Driver
    ecm_cuda.exe used to produce a real stage-1 save.  Default:
    <repo>\build_cuda_cmake\ecm_cuda.exe
#>
param(
    [string]$Exe = '',
    [string]$Ref = '',
    [string]$Driver = '',
    [string]$Sandbox = ''
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
if (-not $Exe) { $Exe = Join-Path $repoRoot 'build_cuda_cmake\stage2_tree_ref.exe' }
if (-not $Ref) { $Ref = Join-Path $repoRoot 'build_cuda_cmake\stage2_ref.exe' }
if (-not $Driver) { $Driver = Join-Path $repoRoot 'build_cuda_cmake\ecm_cuda.exe' }

# A host-only reference with no CMake target: missing means "not built yet", which is a
# skip, not a failure (build it with tools\build\test\build_stage2_tree_ref.ps1).
if (-not (Test-Path $Exe)) {
    Write-Host ("SKIP: stage2_tree_ref.exe not found at " + $Exe +
                " (build it with tools\build\test\build_stage2_tree_ref.ps1)")
    exit 0
}
if (-not $Sandbox) { $Sandbox = Join-Path $repoRoot ('tools\test\_run\stage2tree_' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Force -Path $Sandbox | Out-Null

# GMP DLL next to the tools
$gmpDll = Join-Path $repoRoot 'third_party\gmp-zen3\dist\bin\gmp-10.dll'
if (Test-Path $gmpDll) {
    foreach ($dir in @((Split-Path -Parent $Exe), (Split-Path -Parent $Ref),
                       (Split-Path -Parent $Driver))) {
        if ((Test-Path $dir) -and -not (Test-Path (Join-Path $dir 'gmp-10.dll'))) {
            Copy-Item $gmpDll $dir -Force
        }
    }
}

$N128 = '340282366920938463463374607431768211457'   # 2^128+1 = 59649589127497217 * 5704689200685129054721
$F17 = '59649589127497217'

function Factors([string]$output) {
    $m = [regex]::Match($output, 'stage2: algorithm=tree .*?(?<!bad_)factors=([^ ]*)')
    if (-not $m.Success) { return $null }
    $f = $m.Groups[1].Value
    if ([string]::IsNullOrEmpty($f)) { return @() }
    return @($f -split ',' | Sort-Object)
}

Write-Host "stage2_tree_ref acceptance"
Write-Host ("exe     : " + $Exe)
Write-Host ("ref     : " + $Ref)
Write-Host ("driver  : " + $Driver)
Write-Host ("sandbox : " + $Sandbox)
Write-Host ""

# ---------------------------------------------------------------------------------
Write-Host "[1] the tool's own selftest (polynomial core oracles + frozen known-factor case)"
$out = & $Exe --selftest 2>&1 | Out-String
$code = $LASTEXITCODE
$m = [regex]::Match($out, 'selftest: (\d+) checks, (\d+) failed')
Check "selftest ran" ($m.Success) "no summary line"
if ($m.Success) {
    $checks = [int]$m.Groups[1].Value
    $failed = [int]$m.Groups[2].Value
    Check "selftest reports at least 10 checks" ($checks -ge 10) ("checks=" + $checks)
    Check "no selftest check failed" ($failed -eq 0) ("failed=" + $failed)
    Check "selftest exit code is 0" ($code -eq 0) ("exit=" + $code)
}
Check "poly_mul is checked against pointwise evaluation" ($out -match 'poly_mul == pointwise evaluation')
Check "poly_divmod is checked by reconstructing q\*b + r" ($out -match 'poly_divmod: a == q\*b \+ r')
Check "F vanishes at every root (independent invariant)" ($out -match 'F\(x_j\) == 0 at every root')
Check "the remainder tree is checked against Horner over a PRIME modulus" `
      ($out -match 'remainder tree == Horner at every giant point \(prime modulus\)')
Check "the remainder tree is checked against Horner over a COMPOSITE modulus" `
      ($out -match 'remainder tree == Horner at every giant point \(COMPOSITE modulus')
Check "the frozen known factor is found by the tree" ($out -match 'tree finds a factor of 2\^128\+1')
Check "the selftest's own naive check has no mismatch" `
      ($out -match 'the naive check of that run has no mismatch')

# ---------------------------------------------------------------------------------
Write-Host "[2] frozen vector: N = 2^128+1, sigma=26, B1=1e3, B2=1e6, D=210"
$frozen = (& $Exe --n $N128 --sigma 26 --b1 1000 --b2 1000000 --d 210 2>&1 | Out-String)
$sm = [regex]::Match($frozen,
    'stage2: algorithm=tree curves=(\d+) hits=(\d+) bad_factors=(\d+) factors=([^ ]*) hit_primes=([^ ]*) elapsed=([0-9.]+) operand_bits=(\d+)')
Check "tree summary line present" ($sm.Success) $frozen.Trim()
$treeFactors = Factors $frozen
if ($sm.Success) {
    Check "one curve was processed" ([int]$sm.Groups[1].Value -eq 1) ("curves=" + $sm.Groups[1].Value)
    Check "the tree found the 17-digit factor 59649589127497217" `
          ($sm.Groups[4].Value -match [regex]::Escape($F17)) ("factors=" + $sm.Groups[4].Value)
    Check "no bogus factor (bad_factors = 0)" ([int]$sm.Groups[3].Value -eq 0) `
          ("bad_factors=" + $sm.Groups[3].Value)
    Check "the hit is attributed to stage-2 prime 114713" ($sm.Groups[5].Value -match '114713') `
          ("hit_primes=" + $sm.Groups[5].Value)
    Check "operand_bits is reported and non-zero" ([uint64]$sm.Groups[7].Value -gt 0) `
          ("operand_bits=" + $sm.Groups[7].Value)
    Check "the shape line reports baby_j=24 = phi(210)/2" ($frozen -match 'baby_j=24 ')
    Check "F_degree equals the baby count" ($frozen -match 'F_degree=24')
}

# factor-set equality with the pairing reference (the acceptance criterion)
if (Test-Path $Ref) {
    $pairOut = (& $Ref --n $N128 --sigma 26 --b1 1000 --b2 1000000 --d 210 --algorithm pairing 2>&1 | Out-String)
    $pm = [regex]::Match($pairOut,
        'stage2: algorithm=pairing curves=(\d+) hits=(\d+) bad_factors=(\d+) factors=([^ ]*) hit_primes=([^ ]*)')
    Check "pairing reference summary line present" ($pm.Success) $pairOut.Trim()
    if ($pm.Success -and $null -ne $treeFactors) {
        $pairFactors = if ($pm.Groups[4].Value) { @($pm.Groups[4].Value -split ',' | Sort-Object) } else { @() }
        $same = (($treeFactors -join ',') -eq ($pairFactors -join ','))
        Check "the tree's factor set EQUALS the pairing reference's factor set" $same `
              ("tree=[" + ($treeFactors -join ',') + "] pairing=[" + ($pairFactors -join ',') + "]")
        Check "the two algorithms name the same stage-2 prime" `
              ($pm.Groups[5].Value -match '114713') ("pairing hit_primes=" + $pm.Groups[5].Value)
        Check "the pairing reference reports no bogus factor" ([int]$pm.Groups[3].Value -eq 0)
    }
} else {
    Write-Host ("  [skip] pairing reference not found at " + $Ref +
                " -- factor-set equality not checked")
}

# ---------------------------------------------------------------------------------
Write-Host "[3] --naive-check: remainder tree vs direct Horner, two different shapes"
$shapes = @(
    @{ b1 = 1000; b2 = 1000000; d = 210 },
    @{ b1 = 1000; b2 = 200000;  d = 462 }
)
foreach ($s in $shapes) {
    $o = (& $Exe --n $N128 --sigma 26 --b1 $s.b1 --b2 $s.b2 --d $s.d --naive-check 2>&1 | Out-String)
    $nm = [regex]::Match($o, 'naive_check: points=(\d+) mismatches=(\d+)')
    $tag = ("B1=" + $s.b1 + " B2=" + $s.b2 + " D=" + $s.d)
    Check ("naive check ran and visited every giant point (" + $tag + ")") `
          ($nm.Success -and [int]$nm.Groups[1].Value -gt 0) $o.Trim()
    if ($nm.Success) {
        Check ("naive check has zero mismatches (" + $tag + ")") `
              ([int]$nm.Groups[2].Value -eq 0) ("mismatches=" + $nm.Groups[2].Value)
    }
    Check ("naive-check run reports no bogus factor (" + $tag + ")") `
          ($o -match 'bad_factors=0')
    Check ("naive-check run exits 0 (" + $tag + ")") ($LASTEXITCODE -eq 0) ("exit=" + $LASTEXITCODE)
}

# ---------------------------------------------------------------------------------
Write-Host "[4] sharpness: B2 = 114000 must find NOTHING (largest group-order prime is 114713)"
$sharp = (& $Exe --n $N128 --sigma 26 --b1 1000 --b2 114000 --d 210 2>&1 | Out-String)
$shm = [regex]::Match($sharp, 'stage2: algorithm=tree curves=(\d+) hits=(\d+) bad_factors=(\d+) factors=([^ ]*)')
Check "sharpness run produced a summary line" ($shm.Success) $sharp.Trim()
if ($shm.Success) {
    Check "hits = 0" ([int]$shm.Groups[2].Value -eq 0) ("hits=" + $shm.Groups[2].Value)
    Check "the factor list is empty" ([string]::IsNullOrEmpty($shm.Groups[4].Value)) `
          ("factors=" + $shm.Groups[4].Value)
    Check "bad_factors = 0" ([int]$shm.Groups[3].Value -eq 0)
}
if (Test-Path $Ref) {
    $rp = (& $Ref --n $N128 --sigma 26 --b1 1000 --b2 114000 --d 210 --algorithm pairing 2>&1 | Out-String)
    $rpm = [regex]::Match($rp, 'stage2: algorithm=pairing .*?bad_factors=(\d+) factors=([^ ]*)')
    Check "the pairing reference agrees (finds nothing either)" `
          ($rpm.Success -and [string]::IsNullOrEmpty($rpm.Groups[2].Value)) $rp.Trim()
}

# ---------------------------------------------------------------------------------
Write-Host "[5] real pipeline: a genuine stage-1 save drives the tree stage 2"
$save = $null
if (Test-Path $Driver) {
    $errFile = Join-Path $Sandbox 'driver.err'
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $driverOut = ($N128 | & $Driver --method mont --exponent lcm -sigma 26 -gpucurves 1 `
        --tmp-dir $Sandbox 1000 2>$errFile | Out-String)
    $driverExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    Check "driver exit code is 0" ($driverExit -eq 0) ("exit=" + $driverExit)
    $dm = [regex]::Match($driverOut, 'Saved \d+ Montgomery curve line\(s\) to (\S+)')
    Check "the driver reports where it saved the curve" ($dm.Success) $driverOut.Trim()
    if ($dm.Success) { $save = $dm.Groups[1].Value }
} else {
    Write-Host ("  [skip] driver not found at " + $Driver + " -- save-driven run not checked")
}
Check "driver wrote the save file" ($null -ne $save -and (Test-Path $save)) ([string]$save)
if ($null -ne $save -and (Test-Path $save)) {
    $saveOut = (& $Exe --n $N128 --save $save --b2 1000000 --d 210 2>&1 | Out-String)
    $svm = [regex]::Match($saveOut, 'stage2: algorithm=tree curves=(\d+) hits=(\d+) bad_factors=(\d+) factors=([^ ]*)')
    Check "the save-driven run produced a summary line" ($svm.Success) $saveOut.Trim()
    if ($svm.Success) {
        Check "the save yielded at least one curve" ([int]$svm.Groups[1].Value -ge 1) `
              ("curves=" + $svm.Groups[1].Value)
        Check "the save-driven run found the known factor" `
              ($svm.Groups[4].Value -match [regex]::Escape($F17)) ("factors=" + $svm.Groups[4].Value)
        Check "the save-driven run reports no bogus factor" ([int]$svm.Groups[3].Value -eq 0)
        $saveFactors = Factors $saveOut
        if ($null -ne $treeFactors) {
            Check "the save-driven factor set equals the ladder-driven (synthetic) one" `
                  (($saveFactors -join ',') -eq ($treeFactors -join ',')) `
                  ("save=[" + ($saveFactors -join ',') + "] ladder=[" + ($treeFactors -join ',') + "]")
        }
    }
}

# ---------------------------------------------------------------------------------
Write-Host "[6] cost accounting: measured breakdown, model check, batched structure"
$cm = [regex]::Match($frozen,
    'cost: S=(\d+) poly_muls=(\d+) operand_bits=(\d+) f_tree=(\d+) giant_tree=(\d+) remainder=(\d+) top_level=(\d+) coeff_muls=(\d+) max_mul=(\d+)x(\d+)')
Check "the cost breakdown line is present" ($cm.Success) "no 'cost:' line"
if ($cm.Success) {
    $f = [uint64]$cm.Groups[4].Value; $g = [uint64]$cm.Groups[5].Value
    $r = [uint64]$cm.Groups[6].Value; $t = [uint64]$cm.Groups[7].Value
    Check "every product tree and the descent were charged" (($f -gt 0) -and ($g -gt 0) -and ($r -gt 0))
    Check "operand_bits equals the four-way breakdown" `
          ([uint64]$cm.Groups[3].Value -eq ($f + $g + $r + $t)) `
          ("total=" + $cm.Groups[3].Value + " parts=" + ($f + $g + $r + $t))
    Check "the largest multiplication is recorded" `
          ([uint64]$cm.Groups[9].Value -gt 0 -and [uint64]$cm.Groups[10].Value -gt 0) `
          ("max_mul=" + $cm.Groups[9].Value + "x" + $cm.Groups[10].Value)
}
Check "the product-tree model is checked against the measured tree cost" `
      ($frozen -match 'cost_model_check: MATCH')
Check "the batched (Prime95-shaped) structure is accounted" `
      ($frozen -match 'cost_model: tree_convention=ours-balanced' -and
       $frozen -match 'cost_model: tree_convention=plan-script')
Check "the batched accounting reports P = phi(D)/2 and num_polyG" `
      ($frozen -match 'P=phi\(D\)/2=24 ' -and $frozen -match 'num_poly_g=199 ')

# a --model-only run must reproduce the same shape numbers without executing stage 2
$mo = (& $Exe --n $N128 --b2 1000000 --d 210 --model-only 2>&1 | Out-String)
Check "--model-only runs and needs no --sigma/--b1" ($LASTEXITCODE -eq 0) $mo.Trim()
Check "--model-only reports the same num_polyG" ($mo -match 'num_poly_g=199 ')
Check "--model-only reports three tree conventions" `
      ((([regex]::Matches($mo, 'tree_convention=')).Count) -eq 3)

# ---------------------------------------------------------------------------------
Write-Host "[7] soundness on a larger shape (bigger D, different baby set)"
$big = (& $Exe --n $N128 --sigma 26 --b1 1000 --b2 400000 --d 2310 --naive-check 2>&1 | Out-String)
$bm = [regex]::Match($big, 'stage2: algorithm=tree curves=(\d+) hits=(\d+) bad_factors=(\d+) factors=([^ ]*)')
Check "the D=2310 shape produced a summary line" ($bm.Success) $big.Trim()
if ($bm.Success) {
    Check "no bogus factor on D=2310 (every reported factor divides N)" `
          ([int]$bm.Groups[3].Value -eq 0) ("bad_factors=" + $bm.Groups[3].Value)
    Check "the baby set is phi(2310)/2 = 240" ($big -match 'baby_j=240 ')
}
$nmBig = [regex]::Match($big, 'naive_check: points=(\d+) mismatches=(\d+)')
Check "the D=2310 naive check has zero mismatches" `
      ($nmBig.Success -and [int]$nmBig.Groups[2].Value -eq 0) $big.Trim()

# ---------------------------------------------------------------------------------
Write-Host "[8] CLI hygiene"
$noN = (& $Exe --b1 1000 --b2 2000 --d 210 2>&1 | Out-String)
Check "missing --n is a non-zero exit" ($LASTEXITCODE -ne 0) ("exit=" + $LASTEXITCODE)
Check "missing --n is reported" ($noN -match '--n <decimal> is required')
$one = (& $Exe --n $N128 --b1 1000 --b2 2000 --d 210 --sigma 26 --curves 1 2>&1 | Out-String)
Check "exactly one summary line is printed per run" `
      ((([regex]::Matches($one, 'stage2: algorithm=')).Count) -eq 1) $one.Trim()
Check "the summary line says algorithm=tree" ($one -match 'stage2: algorithm=tree')

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
