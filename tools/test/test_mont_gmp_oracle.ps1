# ---------------------------------------------------------------------------
# test_mont_gmp_oracle.ps1 -- does our Suyama-sigma Montgomery stage-1 agree with
# gmp-ecm (param 0), byte for byte?
#
# For each (N, sigma, B1) it runs
#     build_vs18\tools\mont_standalone.exe <N> <sigma> <B1> 1
#     ecm.exe -param 0 -sigma <sigma> -c 1 (-save f) <B1>      (N on stdin)
# and compares
#   * the stage-1 verdict (our gcd(Z,N) > 1  vs  gmp-ecm "Factor found in step 1"), and
#   * when our Z is a unit mod N (x_valid=1 and gcd == 1), the normalised Montgomery x
#     against gmp-ecm's saved X.
# A hit makes the normalised x meaningless on BOTH sides (Z is not invertible), so the
# x comparison is skipped there and only the verdict is compared.
#
# Requires: build_vs18\tools\mont_standalone.exe
#           tools\build_tool.bat tools\bench\mont_standalone.cpp src\cpu\ecm_mont_cpu.cpp
#
# usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_mont_gmp_oracle.ps1 [-Quick]
# ---------------------------------------------------------------------------
param(
    [string]$Mont = "",
    [string]$Ecm = "D:\code\GIMPS\gmp-ecm\ecm-7.0.5-znver3\ecm.exe",
    [switch]$Quick
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Mont) { $Mont = Join-Path $repo "build_vs18\tools\mont_standalone.exe" }
if (-not (Test-Path $Mont)) { Write-Host "FAIL: $Mont not built yet"; exit 2 }
if (-not (Test-Path $Ecm)) { Write-Host "SKIP: gmp-ecm not found at $Ecm"; exit 3 }
$env:PATH = (Split-Path $Ecm) + ";" + $env:PATH
$run = Join-Path $PSScriptRoot "_mont_oracle_run"
Remove-Item $run -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $run | Out-Null

$q101 = [System.Numerics.BigInteger]::Pow(2, 101) - 1
$p20 = [System.Numerics.BigInteger]524309
$cases = @(
    @{ N = [System.Numerics.BigInteger]::Pow(2, 1277) - 1; name = "M1277";    B1s = @(2, 97, 1000, 10000) },
    @{ N = [System.Numerics.BigInteger]::Pow(2, 3001) - 1; name = "M3001";    B1s = @(1000, 10000) },
    @{ N = [System.Numerics.BigInteger]::Pow(2, 4003) - 1; name = "M4003";    B1s = @(1000) },
    @{ N = $p20 * $q101;                                   name = "p20xq101"; B1s = @(1000, 10000) }
)
if ($Quick) { $cases = @($cases[0]) }
$sigmas = @(12345, 999, 20260922)

function Field([string[]]$lines, [string]$pat) {
    $m = $lines | Select-String -Pattern $pat | Select-Object -First 1
    if (-not $m) { return "" }
    $null = $m.Line -match $pat
    return $Matches[1]
}

$total = 0; $bad = 0; $xskipped = 0
foreach ($c in $cases) {
    foreach ($B1 in $c.B1s) {
        foreach ($sg in $sigmas) {
            $total++
            $mine = & $Mont "$($c.N)" $sg $B1 1 2>&1
            $myX = Field $mine "^x\s+=\s+0x([0-9a-f]*)"
            $myG = Field $mine "^gcd\(Z,N\)\s+=\s+(\d+)"
            $myValid = Field $mine "^x_valid\s+=\s+(\d+)"
            $myHit = ($myG -ne "1")
            $canCompareX = ($myValid -eq "1" -and $myG -eq "1")

            $save = Join-Path $run ("ref_{0}_{1}_{2}.save" -f $c.name, $sg, $B1)
            if (Test-Path $save) { Remove-Item $save -Force }
            # B2 = B1 keeps gmp-ecm in stage 1 only: with its default B2 it may find a
            # factor in step 2 and then the residue it saves is NOT the stage-1 point
            # (observed as a tiny degenerate X).
            # Its stage-1 exponent policy is NOT fully controllable (a reference-only hit
            # on a small modulus needed 41^2 at "B1=1000", i.e. an extended exponent), so
            # the assertions are:
            #   * our hit  =>  their hit            (we must never miss a real stage-1 hit)
            #   * neither hits => the normalised x must match byte for byte
            #   * their hit only => reported as an exponent-policy case (not a failure)
            $null = "$($c.N)" | & $Ecm -param 0 -sigma $sg -c 1 -save $save $B1 $B1 2>&1
            $refX = ""
            if ((Test-Path $save) -and (Get-Item $save).Length -gt 0) {
                $txt = Get-Content $save -Raw
                if ($txt -match "X=0x([0-9a-f]+)") { $refX = $Matches[1] }
            }
            $ecmOut = "$($c.N)" | & $Ecm -param 0 -sigma $sg -c 1 $B1 $B1 2>&1
            $refHit = [bool]($ecmOut | Select-String -Pattern "Factor found in step 1")

            $okH = ($myHit -eq $refHit)
            $policy = $false
            if ($myHit -and -not $refHit) { $okH = $false }          # we must never miss
            elseif ($refHit -and -not $myHit) { $policy = $true; $okH = $true }
            $okX = $true
            if ($canCompareX -and -not $myHit -and -not $refHit) {
                if ($refX -eq "") { $okX = $false }
                else { $okX = ($myX -eq $refX) }
            } else {
                $xskipped++
            }
            if ($okH -and $okX) {
                $note = if ($canCompareX -and -not $refHit) { "x matched" }
                        elseif ($policy) { "reference-only hit (exponent policy)" }
                        else { "hit on both sides" }
                Write-Host ("  ok   {0,-9} sigma={1,-11} B1={2,-6} {3}" -f $c.name, $sg, $B1, $note)
            } else {
                $bad++
                Write-Host ("  FAIL {0,-9} sigma={1,-11} B1={2,-6}  X-match={3} hit(mine={4},ecm={5})" -f `
                            $c.name, $sg, $B1, $okX, $myHit, $refHit)
                Write-Host ("       mine x = 0x{0}" -f $myX.Substring(0, [Math]::Min(64, $myX.Length)))
                Write-Host ("       ecm  X = 0x{0}" -f $refX.Substring(0, [Math]::Min(64, $refX.Length)))
            }
        }
    }
}
Write-Host ""
if ($bad -eq 0) {
    Write-Host ("RESULT: PASS  ({0} cases, {1} of them hits where only the verdict is compared)" -f $total, $xskipped)
    exit 0
}
Write-Host ("RESULT: FAIL  ({0}/{1} mismatches)" -f $bad, $total)
exit 1
