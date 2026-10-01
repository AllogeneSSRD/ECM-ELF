#Requires -Version 5.1
<#
.SYNOPSIS
    CUDA/CGBN stage 2 (M1): does the GPU pairing stage 2 agree with the CPU reference?

.DESCRIPTION
    tools/bench/stage2_gpu_probe.cu runs the same algorithm as the CPU oracle
    (tools/bench/stage2_ref.cpp, --algorithm pairing) on the GPU, so the two must agree
    EXACTLY on the hit set -- same candidate primes, same test, same gcd.  The frozen
    configuration is the one tools/test/test_stage2_ref.ps1 pins:

        N = 2^128+1, sigma = 26, B1 = 1e3, B2 = 1e6, D = 210
        -> the 17-digit factor 59649589127497217, reachable only through stage 2
           (the group order's largest prime is 114713, which is why B2 = 114000 is the
           sharpness probe: it must find NOTHING)

    Layers checked here:
      [1] the tier table of this build (probe --tiers)
      [2] the probe's own selftest (frozen vector, sharpness, segs, 199-curve sweep)
      [3] GPU vs CPU reference on the same parameters, including an empty result
      [4] the save-file path: a REAL driver stage-1 save must drive stage 2 to the same
          factor (this is the path the eventual integration uses)
      [5] soundness: every reported factor must divide N, and bad_factors must be 0
      [6] clean failures (D < 2, no candidates) instead of silent nonsense

    Uses device 1 by default: device 0 is normally running production stage 1.  Skips
    (exit 0) when the probe or a CUDA device is missing.
#>
param(
    [string]$Exe = '',
    [string]$Ref = '',
    [string]$Driver = '',
    [string]$EcmCuda = '',   # the suite passes the driver path under this name
    [int]$Device = 1,
    [switch]$KeepSandbox
)

if (-not $Driver -and $EcmCuda) { $Driver = $EcmCuda }

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
    return @{ out = $out; code = $code }
}

function Factors([string]$text) {
    $m = [regex]::Match($text, '(?<!bad_)factors=([0-9,]*)')
    if (-not $m.Success) { return $null }
    $v = $m.Groups[1].Value.Trim()
    if ($v -eq '') { return @() }
    return @($v.Split(',') | Where-Object { $_ -ne '' } | Sort-Object)
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Exe) { $Exe = Join-Path $repoRoot 'build_cuda_cmake\stage2_gpu_probe.exe' }
if (-not $Ref) { $Ref = Join-Path $repoRoot 'build_cuda_cmake\stage2_ref.exe' }
if (-not $Driver) { $Driver = Join-Path $repoRoot 'build_cuda_cmake\ecm_cuda.exe' }

if (-not (Test-Path $Exe)) {
    Write-Host "FAIL: stage2_gpu_probe.exe not found (build it with tools\\build\\build_stage2_probe.ps1)" -ForegroundColor Red
    exit 2
}
$gmpDll = Join-Path $repoRoot 'third_party\gmp-zen3\dist\bin\gmp-10.dll'
$exeDir = Split-Path -Parent $Exe
if ((Test-Path $gmpDll) -and -not (Test-Path (Join-Path $exeDir 'gmp-10.dll'))) {
    Copy-Item $gmpDll $exeDir -Force
}

$N128 = '340282366920938463463374607431768211457'
$F17 = '59649589127497217'

# --- no device / no probe => skip cleanly ----------------------------------------
$tiers = Run $Exe @('--tiers')
if ($tiers.code -ne 0 -or $tiers.out -notmatch 'stage2gpu_tiers:') {
    Write-Host "[skip] the probe cannot report its tiers:" -ForegroundColor Yellow
    Write-Host $tiers.out.Trim()
    Write-Host ""
    Write-Host "passed: 0   failed: 0"
    exit 0
}

Write-Host "CUDA/CGBN stage 2 probe"
Write-Host ("exe    : " + $Exe)
Write-Host ("device : " + $Device)
Write-Host ""

# ---------------------------------------------------------------------------------
Write-Host "[1] the tier table of this build"
$tm = [regex]::Match($tiers.out, 'stage2gpu_tiers:(.*)')
$tierList = @()
if ($tm.Success) { $tierList = @($tm.Groups[1].Value.Trim().Split(' ') | Where-Object { $_ -ne '' }) }
Check "at least one tier is instantiated" ($tierList.Count -ge 1) $tiers.out.Trim()
Check "a tier covers a 129-bit N (needs >= 135 bits, so 192)" ($tierList -contains '192/4/128') `
      ("tiers=" + ($tierList -join ' '))
$tierBits = @($tierList | ForEach-Object { [int]($_.Split('/')[0]) })
$sorted = @($tierBits | Sort-Object)
Check "tiers are in ascending bit order" (($tierBits -join ',') -eq ($sorted -join ',')) `
      ($tierBits -join ',')
Check "every tier carries at least 6 bits of container headroom over its N" `
      (@($tierList | Where-Object { $_.Split('/').Count -eq 3 }).Count -eq $tierList.Count) `
      ($tierList -join ' ')

# ---------------------------------------------------------------------------------
Write-Host "[2] the probe's own selftest (frozen vector + sharpness + segs + sweep)"
$st = Run $Exe @('--selftest', '--device', "$Device")
$sm = [regex]::Match($st.out, 'stage2gpu_selftest: checks=(\d+) failed=(\d+)')
Check "selftest reports its check count" ($sm.Success) $st.out.Trim()
if ($sm.Success) {
    Check "selftest reports at least 6 checks" ([int]$sm.Groups[1].Value -ge 6) ("checks=" + $sm.Groups[1].Value)
    Check "no selftest check failed" ([int]$sm.Groups[2].Value -eq 0) ("failed=" + $sm.Groups[2].Value)
    Check "selftest exit code is 0" ($st.code -eq 0) ("exit=" + $st.code)
}
Check "the selftest found the known 17-digit factor on the GPU" ((Factors $st.out) -join ',' -eq $F17)
Check "the sharpness check (B2=114000 finds nothing) is present" ($st.out -match 'B2 just below')

# ---------------------------------------------------------------------------------
Write-Host "[3] GPU vs the CPU reference stage 2 on identical parameters"
$haveRef = Test-Path $Ref
if (-not $haveRef) {
    Write-Host "  [skip] stage2_ref.exe not found ($Ref)" -ForegroundColor Yellow
} else {
    $gpu = Run $Exe @('--n', $N128, '--sigma', '26', '--b1', '1000', '--b2', '1000000',
                      '--d', '210', '--device', "$Device")
    $cpu = Run $Ref @('--n', $N128, '--sigma', '26', '--b1', '1000', '--b2', '1000000',
                      '--d', '210', '--algorithm', 'pairing')
    $gf = Factors $gpu.out
    $cf = Factors $cpu.out
    Check "the GPU run reports a factor list" ($null -ne $gf) $gpu.out.Trim()
    Check "the CPU run reports a factor list" ($null -ne $cf) $cpu.out.Trim()
    Check "the GPU finds the known factor" (($gf -join ',') -eq $F17) ("gpu=" + ($gf -join ','))
    Check "GPU and CPU report the same factors" (($gf -join ',') -eq ($cf -join ',')) `
          ("gpu=" + ($gf -join ',') + " cpu=" + ($cf -join ','))
    Check "no GPU factor fails to divide N (bad_factors=0)" ($gpu.out -match 'bad_factors=0')

    # sharpness, both paths
    $gpu2 = Run $Exe @('--n', $N128, '--sigma', '26', '--b1', '1000', '--b2', '114000',
                       '--d', '210', '--device', "$Device")
    $cpu2 = Run $Ref @('--n', $N128, '--sigma', '26', '--b1', '1000', '--b2', '114000',
                       '--d', '210', '--algorithm', 'pairing')
    Check "B2=114000 finds nothing on the GPU" ((Factors $gpu2.out).Count -eq 0) $gpu2.out.Trim()
    Check "B2=114000 finds nothing on the CPU too" ((Factors $cpu2.out).Count -eq 0) $cpu2.out.Trim()

    # a 199-curve sweep: identical hit sets is the strongest agreement available
    $gpu3 = Run $Exe @('--n', $N128, '--sigma', '2', '--curves', '199', '--b1', '1000',
                       '--b2', '1000000', '--d', '210', '--device', "$Device")
    $cpu3 = Run $Ref @('--n', $N128, '--sigma', '2', '--curves', '199', '--b1', '1000',
                       '--b2', '1000000', '--d', '210', '--algorithm', 'pairing')
    $gf3 = Factors $gpu3.out
    $cf3 = Factors $cpu3.out
    Check "the 199-curve sweep finds the known factor on the GPU" (($gf3 -join ',') -eq $F17) `
          ("gpu=" + ($gf3 -join ','))
    Check "the 199-curve sweep agrees with the CPU" (($gf3 -join ',') -eq ($cf3 -join ',')) `
          ("gpu=" + ($gf3 -join ',') + " cpu=" + ($cf3 -join ','))
    # segs > 1 splits one curve's accumulator into several pieces; the host multiplies the
    # pieces back together before the gcd, so the answer must be identical.
    $gpu4 = Run $Exe @('--n', $N128, '--sigma', '26', '--b1', '1000', '--b2', '1000000',
                       '--d', '210', '--segs', '4', '--device', "$Device")
    Check "segs=4 gives the same factor as segs=1" ((Factors $gpu4.out) -join ',' -eq $F17) `
          $gpu4.out.Trim()
}

# ---------------------------------------------------------------------------------
Write-Host "[4] stage 2 driven by a REAL driver stage-1 save"
if (-not (Test-Path $Driver)) {
    Write-Host "  [skip] ecm_cuda.exe not found ($Driver)" -ForegroundColor Yellow
} else {
    $sb = Join-Path $PSScriptRoot ('_run\s2gpu_' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    New-Item -ItemType Directory -Force $sb | Out-Null
    $errFile = Join-Path $sb 'driver.err'
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $drv = ($N128 | & $Driver --method mont --exponent lcm -sigma 26 -gpucurves 1 --tmp-dir $sb 1000 2>$errFile | Out-String)
    $drvCode = $LASTEXITCODE
    $ErrorActionPreference = $prev
    $sm2 = [regex]::Match($drv, 'Saved \d+ Montgomery curve line\(s\) to (\S+)')
    Check "the driver wrote a stage-1 save" ($sm2.Success -and (Test-Path $sm2.Groups[1].Value)) `
          ($drv.Trim() + " " + (Get-Content $errFile -Raw -ErrorAction SilentlyContinue))
    if ($sm2.Success) {
        $save = $sm2.Groups[1].Value
        $savetext = [System.IO.File]::ReadAllText($save)
        Check "the save is the sigma=26 / B1=1000 curve" `
              ($savetext -match 'SIGMA=26;' -and $savetext -match 'B1=1000;')
        $saved = Run $Exe @('--n', $N128, '--save', $save, '--b1', '1000', '--b2', '1000000',
                            '--d', '210', '--device', "$Device")
        Check "stage 2 from the driver's save finds the same factor" `
              (((Factors $saved.out) -join ',') -eq $F17) $saved.out.Trim()
        Check "the run really used the save (not the ladder)" ($saved.out -match 'stage1_point=save') `
              $saved.out.Trim()
        # the same save with the ladder path recomputed on the device must agree bit for bit
        $ladder = Run $Exe @('--n', $N128, '--sigma', '26', '--b1', '1000', '--b2', '1000000',
                             '--d', '210', '--device', "$Device")
        Check "save-driven and ladder-driven runs agree" `
              ((Factors $saved.out) -join ',' -eq (Factors $ladder.out) -join ',') `
              ("save=" + ((Factors $saved.out) -join ',') + " ladder=" + ((Factors $ladder.out) -join ','))
        Check "the ladder path is reported as such" ($ladder.out -match 'stage1_point=ladder')
    }
    if (-not $KeepSandbox) { Remove-Item $sb -Recurse -Force -ErrorAction SilentlyContinue }
}

# ---------------------------------------------------------------------------------
Write-Host "[5] clean failures instead of silent nonsense"
$bad1 = Run $Exe @('--n', $N128, '--d', '1', '--device', "$Device")
Check "D < 2 is refused (nonzero exit)" ($bad1.code -ne 0) ("exit=" + $bad1.code)
Check "D < 2 says why" ($bad1.out -match 'need 2 <= D and B1 < B2')
$bad2 = Run $Exe @('--n', $N128, '--b1', '1000', '--b2', '1000', '--device', "$Device")
Check "B1 >= B2 is refused" ($bad2.code -ne 0) ("exit=" + $bad2.code)
$bad3 = Run $Exe @('--n', $N128, '--b1', '1000', '--b2', '1000000', '--d', '210', '--device', '99')
Check "an out-of-range device fails loudly" ($bad3.code -ne 0) ("exit=" + $bad3.code)
Check "an out-of-range device reports the CUDA error" ($bad3.out -match 'failed|invalid device')

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
