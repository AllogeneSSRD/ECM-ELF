#Requires -Version 5.1
<#
.SYNOPSIS
    --go group-order diagnostics: param 0 support and the minimum-B1/B2 derivation.

.DESCRIPTION
    The driver's --go mode asks gp/PARI for #E(F_p) of the curve a given sigma produces, so an
    operator can see WHY a curve did or did not find a factor.  Two things are checked here:

      [1] it uses the parametrization of the run that actually executed.  The CPU Montgomery
          engine (and the Edwards one) use Suyama sigma = param 0 -- their saves carry no PARAM=
          key -- while only the GPU engines use gpu_param (0/2/3).  Before this was fixed --go
          reported the order of a PARAM-3 curve for a param-0 run, i.e. a different curve
          entirely (measured: largest_prime 2666737705477 instead of 114713).
      [2] the order's exact factorization is turned into the minimum bounds that would find the
          factor: min_b1_stage1 (stage 1 alone) and min_b1_stage2/min_b2_stage2 (let stage 2
          finish).  That is the deterministic way to build a test vector for a chosen factor.

    Fixture: N = 2^128+1, sigma 26 -- the SAME curve our reference stage 2 (tools/bench/
    stage2_ref.cpp) independently found a factor on.  For this curve #E(F_59649589127497217) =
    2^3*3*7*67*233*331*599*114713, so the derived bounds are B1 >= 599, B2 >= 114713 -- and the
    test then PROVES the bounds are real and sharp by running the reference stage 2 with exactly
    those bounds (it must find the factor) and with B2 one prime short (it must not).

    Needs gp/PARI (skipped with a clear message when it is missing).
#>
param(
    [string]$Driver = '',
    [string]$Ref = '',
    [string]$Gp = ''
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
if (-not $Driver) { $Driver = Join-Path $repoRoot 'build_cuda_cmake\ecm_cuda.exe' }
if (-not $Ref) { $Ref = Join-Path $repoRoot 'build_cuda_cmake\stage2_ref.exe' }
if (-not (Test-Path $Driver)) { Write-Host "FAIL: ecm_cuda.exe not found (pass -Driver <path>)" -ForegroundColor Red; exit 2 }
if (-not (Test-Path $Ref)) { Write-Host "FAIL: stage2_ref.exe not found (pass -Ref <path>)" -ForegroundColor Red; exit 2 }
if (-not $Gp) {
    # Get-Command does not always see gp even when cmd's PATH does, so try several ways.
    $cand = Get-Command 'gp.exe' -ErrorAction SilentlyContinue
    if ($cand) { $Gp = $cand.Source }
    if (-not $Gp) {
        $w = (& cmd /c "where gp 2>nul" | Select-Object -First 1)
        if ($w -and (Test-Path $w)) { $Gp = $w }
    }
    if (-not $Gp) {
        $dirs = @('D:\AppData', 'C:\Program Files', 'C:\Program Files (x86)')
        foreach ($d in $dirs) {
            if (-not (Test-Path $d)) { continue }
            $hit = Get-ChildItem $d -Directory -Filter 'Pari*' -ErrorAction SilentlyContinue |
                   ForEach-Object { Join-Path $_.FullName 'gp.exe' } |
                   Where-Object { Test-Path $_ } | Select-Object -First 1
            if ($hit) { $Gp = $hit; break }
        }
    }
}
if (-not $Gp -or -not (Test-Path $Gp)) {
    Write-Host "[skip] gp/PARI not found -- install it or pass -Gp <path\\gp.exe>" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "passed: 0   failed: 0"
    exit 0
}

$N128 = '340282366920938463463374607431768211457'      # 2^128+1
$F17 = '59649589127497217'
$Sigma = 26
$B1 = 1000000                                          # >= 114713, so stage 1 alone finds it

Write-Host "--go group-order diagnostics"
Write-Host ("driver  : " + $Driver)
Write-Host ("ref     : " + $Ref)
Write-Host ("gp      : " + $Gp)
Write-Host ""

# ---------------------------------------------------------------------------------
Write-Host "[1] --go reports the PARAM 0 curve (the one --method mont actually ran)"
$prev = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$out = ($N128 | & $Driver --method mont --exponent lcm -sigma $Sigma -gpucurves 1 --go `
        --gp $Gp $B1 2>&1 | Out-String)
$ErrorActionPreference = $prev

$hit = [regex]::Match($out, 'factor\[0\]=(\d+)')
Check "stage 1 found the 17-digit factor at B1=1e6" ($hit.Success -and $hit.Groups[1].Value -eq $F17) `
      ($hit.Value)
$mb = [regex]::Match($out, 'go_min_bounds\[0\]: param=(\d+) largest_prime=(\d+) min_b1_stage1=(\d+) min_b1_stage2=(\d+) min_b2_stage2=(\d+)')
Check "the go_min_bounds line is printed" ($mb.Success) ($out -split "`n" | Select-String 'go_min_bounds' | ForEach-Object { $_.Line.Trim() })
if ($mb.Success) {
    Check "it uses param 0 for the CPU Montgomery run" ($mb.Groups[1].Value -eq '0') ("param=" + $mb.Groups[1].Value)
    Check "the largest prime factor of #E is 114713" ($mb.Groups[2].Value -eq '114713') ("largest_prime=" + $mb.Groups[2].Value)
    Check "min_b1_stage1 is that same prime (stage 1 alone)" ($mb.Groups[3].Value -eq '114713') ("min_b1_stage1=" + $mb.Groups[3].Value)
    Check "min_b1_stage2 is the rest of the factorization (599)" ($mb.Groups[4].Value -eq '599') ("min_b1_stage2=" + $mb.Groups[4].Value)
    Check "min_b2_stage2 is the largest prime (114713)" ($mb.Groups[5].Value -eq '114713') ("min_b2_stage2=" + $mb.Groups[5].Value)
}
$gf = [regex]::Match($out, 'go_factor\[0\]=([^\r\n]*)')
Check "the factorization line names 114713" ($gf.Success -and $gf.Groups[1].Value -match '114713') ($gf.Value)
Check "the group order itself is reported" ([regex]::IsMatch($out, 'go\[0\]=\d{15,}'))

# ---------------------------------------------------------------------------------
Write-Host "[2] the derived bounds are real: reference stage 2 with B1=599, B2=114713 finds it"
$prev = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$ok_run = (& $Ref --n $N128 --b1 599 --b2 114713 --d 210 --sigma $Sigma --curves 1 --algorithm pairing 2>&1 | Out-String)
$ErrorActionPreference = $prev
$ok_m = [regex]::Match($ok_run, 'stage2: algorithm=pairing curves=(\d+) hits=(\d+) bad_factors=(\d+) factors=([^ ]*)')
Check "the reference finds the factor with exactly the derived bounds" `
      ($ok_m.Success -and $ok_m.Groups[4].Value -match [regex]::Escape($F17)) ($ok_m.Value.Trim())

Write-Host "[3] and they are sharp: B2 = 114000 (one prime short) must NOT find it"
$prev = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$neg_run = (& $Ref --n $N128 --b1 599 --b2 114000 --d 210 --sigma $Sigma --curves 1 --algorithm pairing 2>&1 | Out-String)
$ErrorActionPreference = $prev
$neg_m = [regex]::Match($neg_run, 'stage2: algorithm=pairing curves=(\d+) hits=(\d+) bad_factors=(\d+) factors=([^ ]*)')
Check "no factor below the derived B2" ($neg_m.Success -and [int]$neg_m.Groups[2].Value -eq 0) ($neg_m.Value.Trim())
Check "and still no bogus factor" ($neg_m.Success -and [int]$neg_m.Groups[3].Value -eq 0)

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
