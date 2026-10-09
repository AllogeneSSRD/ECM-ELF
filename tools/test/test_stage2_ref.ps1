#Requires -Version 5.1
<#
.SYNOPSIS
    stage2_ref acceptance -- the reference ECM stage 2 (algorithm + correctness first).

.DESCRIPTION
    Milestone M1 of docs/architecture/STAGE2.md: prove the stage-2 algorithm and the
    save-file conventions BEFORE any GPU work.  Layers checked here:

      [1] the tool's own --selftest: x-only curve arithmetic (xDBL / xADD / ladder)
          against naive AFFINE arithmetic over a prime field, plus an end-to-end run on
          2^128+1 with a FROZEN deterministic configuration (sigma=26, B1=1e3, B2=1e6,
          D=210 -> the 17-digit factor 59649589127497217 through stage-2 prime 114713).
      [2] convention: the REAL driver (ecm_cuda.exe --method mont, exponent lcm) writes
          a stage-1 save for that same curve; stage2_ref must recompute the SAME
          normalised x from (N, sigma, B1).  This is what ties the reference's curve
          math to our verified stage 1 (which is itself bit-exact against gmp-ecm).
      [3] real pipeline: stage 2 driven by that save file must find the known factor,
          and the two independent algorithms (brute force per prime vs BSGS pairing)
          must agree.
      [4] soundness on a real Mersenne save: every reported factor must divide N.
      [5] CLI hygiene: algorithm selection, missing --n.

    No GPU is needed (everything is CPU/GMP), so this runs in the driver group of the
    suite.  Deterministic: fixed N, sigma, bounds and D.

.PARAMETER Exe
    stage2_ref.exe.  Default: <repo>\build_cuda_cmake\stage2_ref.exe
.PARAMETER Driver
    ecm_cuda.exe used to produce a real stage-1 save.  Default: <repo>\build_cuda_cmake\ecm_cuda.exe
#>
param(
    [string]$Exe = '',
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
if (-not $Exe) { $Exe = Join-Path $repoRoot 'build_cuda_cmake\stage2_ref.exe' }
if (-not $Driver) { $Driver = Join-Path $repoRoot 'build_cuda_cmake\ecm_cuda.exe' }
if (-not (Test-Path $Exe)) { Write-Host "FAIL: stage2_ref.exe not found (pass -Exe <path>)" -ForegroundColor Red; exit 2 }
if (-not (Test-Path $Driver)) { Write-Host "FAIL: ecm_cuda.exe not found (pass -Driver <path>)" -ForegroundColor Red; exit 2 }
if (-not $Sandbox) { $Sandbox = Join-Path $repoRoot ('tools\test\_run\stage2ref_' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Force -Path $Sandbox | Out-Null

# GMP DLL next to the tools
$gmpDll = Join-Path $repoRoot 'third_party\gmp-zen3\dist\bin\gmp-10.dll'
foreach ($dir in @((Split-Path -Parent $Exe), (Split-Path -Parent $Driver))) {
    if ((Test-Path $gmpDll) -and -not (Test-Path (Join-Path $dir 'gmp-10.dll'))) {
        Copy-Item $gmpDll $dir -Force
    }
}

$N128 = '340282366920938463463374607431768211457'      # 2^128+1 = 59649589127497217 * 5704689200685129054721
$F17 = '59649589127497217'

Write-Host "stage2_ref acceptance"
Write-Host ("exe     : " + $Exe)
Write-Host ("driver  : " + $Driver)
Write-Host ("sandbox : " + $Sandbox)
Write-Host ""

# ---------------------------------------------------------------------------------
Write-Host "[1] the tool's own selftest (affine oracle + frozen known-factor case)"
$out = & $Exe --selftest 2>&1 | Out-String
$code = $LASTEXITCODE
$m = [regex]::Match($out, 'selftest: (\d+) checks, (\d+) failed')
Check "selftest ran" ($m.Success) "no summary line"
if ($m.Success) {
    $checks = [int]$m.Groups[1].Value
    $failed = [int]$m.Groups[2].Value
    Check "selftest reports at least 9 checks" ($checks -ge 9) ("checks=" + $checks)
    Check "no selftest check failed" ($failed -eq 0) ("failed=" + $failed)
    Check "selftest exit code is 0" ($code -eq 0) ("exit=" + $code)
    Check "xDBL/xADD/ladder are checked against affine arithmetic" `
          ($out -match 'xDBL matches affine doubling' -and $out -match 'xADD matches affine' -and
           $out -match 'ladder \[k\]P matches affine')
    Check "the known factor of 2^128+1 is found by pairing" `
          ($out -match 'pairing finds a factor of 2\^128\+1')
    Check "both algorithms agree on it" ($out -match 'brute \(independent algorithm\) finds the same')
}

# ---------------------------------------------------------------------------------
Write-Host "[2] convention: stage2_ref reproduces the driver's stage-1 x for the same curve"
# The driver names the stage-1 save itself (<n>_<B1>.save in --tmp-dir); "-save" is for
# factorization lines only.  So take the path from the driver's own output.
$save = $null
# The driver writes progress to stderr; PS 5.1 turns native stderr into a terminating
# error under $ErrorActionPreference = 'Stop', so relax it and redirect stderr to a file.
# -gpucurves is the curve-count flag for EVERY method, including the CPU ones.
$errFile = Join-Path $Sandbox 'driver.err'
$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$driverOut = ($N128 | & $Driver --method mont --exponent lcm -sigma 26 -gpucurves 1 `
    --tmp-dir $Sandbox 1000 2>$errFile | Out-String)
$driverExit = $LASTEXITCODE
$ErrorActionPreference = $prevEap
Check "driver exit code is 0" ($driverExit -eq 0) ("exit=" + $driverExit)
$sm = [regex]::Match($driverOut, 'Saved \d+ Montgomery curve line\(s\) to (\S+)')
Check "the driver reports where it saved the curve" ($sm.Success) $driverOut.Trim()
if ($sm.Success) { $save = $sm.Groups[1].Value }
Check "driver wrote the save file" ($null -ne $save -and (Test-Path $save)) ([string]$save)
if ($null -ne $save -and (Test-Path $save)) {
    $savetext = [System.IO.File]::ReadAllText($save)
    Check "the save is for sigma 26" ($savetext -match 'SIGMA=26;') "no SIGMA=26 in the save"
    Check "the save records B1=1000" ($savetext -match 'B1=1000;') "no B1=1000 in the save"
    $xm = [regex]::Match($savetext, 'X=0x([0-9a-fA-F]+)')
    Check "the save carries an X coordinate" ($xm.Success)
    if ($xm.Success) {
        $saveX = $xm.Groups[1].Value.TrimStart('0').ToLowerInvariant()
        if ($saveX -eq '') { $saveX = '0' }
        $ref = (& $Exe --n $N128 --sigma 26 --b1 1000 --print-stage1-x 2>&1 | Out-String)
        $rm = [regex]::Match($ref, 'x=0x([0-9a-fA-F]+)')
        Check "stage2_ref printed a stage-1 x" ($rm.Success) $ref.Trim()
        if ($rm.Success) {
            $refX = $rm.Groups[1].Value.TrimStart('0').ToLowerInvariant()
            if ($refX -eq '') { $refX = '0' }
            Check "the recomputed x equals the driver's save X" ($refX -eq $saveX) `
                  ("ref=" + $refX.Substring(0, [Math]::Min(24, $refX.Length)) +
                   " save=" + $saveX.Substring(0, [Math]::Min(24, $saveX.Length)))
        }
    }
}

# ---------------------------------------------------------------------------------
Write-Host "[3] real pipeline: stage 2 driven by that save finds the known factor"
if ($null -ne $save -and (Test-Path $save)) {
    $out = (& $Exe --n $N128 --save $save --b2 1000000 --d 210 --algorithm both --verbose-hits 2>&1 | Out-String)
    $pair = [regex]::Match($out, 'stage2: algorithm=pairing curves=(\d+) hits=(\d+) bad_factors=(\d+) factors=([^ ]*)')
    $brute = [regex]::Match($out, 'stage2: algorithm=brute curves=(\d+) hits=(\d+) bad_factors=(\d+) factors=([^ ]*)')
    Check "pairing summary line present" ($pair.Success)
    Check "brute summary line present" ($brute.Success)
    if ($pair.Success -and $brute.Success) {
        Check "the save yielded one parsed curve" ([int]$pair.Groups[1].Value -ge 1) ("curves=" + $pair.Groups[1].Value)
        Check "pairing found the known 17-digit factor" ($pair.Groups[4].Value -match [regex]::Escape($F17)) `
              ("factors=" + $pair.Groups[4].Value)
        Check "brute found it too (independent algorithm)" ($brute.Groups[4].Value -match [regex]::Escape($F17)) `
              ("factors=" + $brute.Groups[4].Value)
        Check "no bogus factor from either algorithm" `
              (([int]$pair.Groups[3].Value -eq 0) -and ([int]$brute.Groups[3].Value -eq 0)) `
              ("pairing bad=" + $pair.Groups[3].Value + " brute bad=" + $brute.Groups[3].Value)
        # the reported hit line must name a prime inside (B1, B2]
        $hm = [regex]::Match($out, 'stage2_hit: sigma=(\d+) factor=(\d+) prime=(\d+)')
        Check "a hit line names the curve and the stage-2 prime" ($hm.Success)
        if ($hm.Success) {
            Check "the hit prime is inside (B1, B2]" `
                  (([uint64]$hm.Groups[3].Value -gt 1000) -and ([uint64]$hm.Groups[3].Value -le 1000000)) `
                  ("prime=" + $hm.Groups[3].Value)
        }
    }
    Check "the save-driven run reports no bogus factor in any line" ($out -notmatch 'bad_factors=[1-9]')
}

# ---------------------------------------------------------------------------------
Write-Host "[4] soundness on a real Mersenne save (M677)"
$m677 = [System.Numerics.BigInteger]::Pow(2, 677) - 1
$ErrorActionPreference = 'Continue'
$mout = ($m677.ToString() | & $Driver --method mont --exponent lcm -sigma 26 -gpucurves 2 `
    --tmp-dir $Sandbox 1000 2>(Join-Path $Sandbox 'driver_m677.err') | Out-String)
$ErrorActionPreference = $prevEap
$msm = [regex]::Match($mout, 'Saved \d+ Montgomery curve line\(s\) to (\S+)')
Check "the driver reports the M677 save" ($msm.Success) $mout.Trim()
$msave = if ($msm.Success) { $msm.Groups[1].Value } else { $null }
Check "driver produced an M677 save" ($null -ne $msave -and (Test-Path $msave)) ([string]$msave)
if ($null -ne $msave -and (Test-Path $msave)) {
    $sout = (& $Exe --n $m677.ToString() --save $msave --b2 200000 --d 210 --algorithm both 2>&1 | Out-String)
    $pr = [regex]::Match($sout, 'stage2: algorithm=pairing curves=(\d+) hits=(\d+) bad_factors=(\d+)')
    $br = [regex]::Match($sout, 'stage2: algorithm=brute curves=(\d+) hits=(\d+) bad_factors=(\d+)')
    Check "the M677 save parsed at least one curve" ($pr.Success -and [int]$pr.Groups[1].Value -ge 1)
    Check "no bogus factor on M677 (pairing)" ($pr.Success -and [int]$pr.Groups[3].Value -eq 0)
    Check "no bogus factor on M677 (brute)" ($br.Success -and [int]$br.Groups[3].Value -eq 0)
    Check "pairing found at least everything brute found (subset relation)" `
          ($pr.Success -and $br.Success -and [int]$br.Groups[2].Value -le [int]$pr.Groups[2].Value) `
          ("brute hits=" + $br.Groups[2].Value + " pairing hits=" + $pr.Groups[2].Value)
}

# ---------------------------------------------------------------------------------
Write-Host "[5] CLI hygiene"
$only = (& $Exe --n $N128 --b1 1000 --b2 2000 --d 210 --sigma 26 --curves 1 --algorithm pairing 2>&1 | Out-String)
Check "algorithm=pairing prints exactly one summary line" `
      ((([regex]::Matches($only, 'stage2: algorithm=')).Count) -eq 1) $only.Trim()
Check "and it is the pairing line" ($only -match 'stage2: algorithm=pairing')
$onlyBrute = (& $Exe --n $N128 --b1 1000 --b2 2000 --d 210 --sigma 26 --curves 1 --algorithm brute 2>&1 | Out-String)
Check "algorithm=brute prints exactly one summary line" `
      ((([regex]::Matches($onlyBrute, 'stage2: algorithm=')).Count) -eq 1) $onlyBrute.Trim()
& $Exe --b1 1000 --b2 2000 2>&1 | Out-Null
Check "missing --n is a non-zero exit" ($LASTEXITCODE -ne 0) ("exit=" + $LASTEXITCODE)

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
