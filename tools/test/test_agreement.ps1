# ---------------------------------------------------------------------------
# test_agreement.ps1 -- three-backend agreement regression for Edwards stage 1.
#
# For the same (N, B1, sigma) it runs three independent implementations:
#     gmp        : scalar mpz path                 (--edwards-backend gmp)
#     simd-mont  : SIMD batch, Montgomery domain   (--edwards-mersenne off)
#     simd-mers  : SIMD batch, Mersenne fold       (--edwards-mersenne on, N = 2^k-1)
# and asserts
#   1) the per-curve verdict agrees: same hit set AND same factor value, and
#   2) the per-curve .tmp archives are byte-identical (same Qx/Qz => the ladder
#      computed exactly the same point).
# (2) is the strong one: it pins three independent implementations to one result,
# and any non-canonical-representation / reduction bug makes some curve drift.
# See docs/ECM_EDWARDS_STAGE1.md new sections: the SIMD CIOS unmasked store and
# the scalar mpn_redc_1 return-value misuse.  The historical failing case
# (M3001 / sigma=20260922 / B1=1e5) is part of the default set, so this script is
# also the regression test for both bugs.
#
# usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/test/test_agreement.ps1
#   ... -File tools/test/test_agreement.ps1 -Exe build_vs18\Release\ecm.exe
#   ... -File tools/test/test_agreement.ps1 -Quick     # fast subset only
# ---------------------------------------------------------------------------
param(
    [string]$Exe = "",
    [switch]$Quick,
    [int]$Curves = 8
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)      # tools/test -> repo
if (-not $Exe) { $Exe = Join-Path $repo "build_vs18\Release\ecm.exe" }
if (-not (Test-Path $Exe)) { Write-Host "FAIL: ecm.exe not found: $Exe"; exit 2 }
$env:PATH = "$repo\third_party\gmp-zen3\dist\bin;$env:PATH"

$work = Join-Path $PSScriptRoot "_agree_run"
Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $work | Out-Null

# Non-Mersenne case: N = 3217073 * (2^521-1), 543 bit.  A 20-bit factor inside a
# big cofactor, so B1=1e5 almost always hits; also checks that auto does NOT pick
# the fold domain for this N.
$q521 = [System.Numerics.BigInteger]::Pow(2, 521) - 1
$nMixed = ([System.Numerics.BigInteger]3217073) * $q521

$allCases = @(
    @{ name = "M3001 B1=2e4 (scalar bug case)"; N = "(2^3001-1)"; B1 = 20000;  sigma = 20260922; backends = @("gmp", "simd-mont", "simd-mers") }
    @{ name = "M3001 B1=1e5 (historic failure)"; N = "(2^3001-1)"; B1 = 100000; sigma = 20260922; backends = @("gmp", "simd-mont", "simd-mers") }
    @{ name = "M677  B1=1e3 (fold sh=51)";       N = "(2^677-1)";  B1 = 1000;   sigma = 11000;    backends = @("gmp", "simd-mont", "simd-mers") }
    @{ name = "M4003 B1=1e3 (fold sh=1)";        N = "(2^4003-1)"; B1 = 1000;   sigma = 9000;     backends = @("gmp", "simd-mont", "simd-mers") }
    @{ name = "non-Mersenne 543bit B1=1e5";      N = "$nMixed";     B1 = 100000; sigma = 20260922; backends = @("gmp", "simd-mont", "simd-auto") }
)
$cases = if ($Quick) { @($allCases | Where-Object { $_.name -match "B1=2e4|sh=1" }) } else { $allCases }

function Run-Backend {
    param([string]$backend, [string]$field, [string]$N, [int]$B1, [int]$sigma, [string]$dir)
    Remove-Item $dir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $argv = @("--edwards", "--edwards-backend", $backend, "--edwards-threads", "1",
              "--edwards-naf-w", "12", "--edwards-mersenne", $field,
              "-gpucurves", "$Curves", "-sigma", "$sigma", "--tmp-dir", $dir, "$B1", "0")
    $out = $N | & $Exe @argv 2>&1
    $fld = ($out | Select-String -Pattern "field\s+:" | Select-Object -First 1)
    $fldName = "?"
    if ($fld) { $fldName = ($fld.Line -replace '.*field\s+:\s*', '').Split(' ')[0] }
    # hit curves -> factor value; comparing the per-curve set is stronger than a count.
    # the driver prefixes every line with a timestamp, so the pattern is unanchored.
    $fact = @{}
    foreach ($m in ($out | Select-String -Pattern "factor\[(\d+)\]=(\d+)")) {
        if ($m.Line -match "factor\[(\d+)\]=(\d+)\s*$") { $fact[[int]$Matches[1]] = $Matches[2] }
    }
    $hashes = @{}
    foreach ($f in (Get-ChildItem $dir -Filter *.tmp -ErrorAction SilentlyContinue | Sort-Object Name)) {
        $hashes[$f.Name] = (Get-FileHash $f.FullName -Algorithm SHA256).Hash
    }
    return [pscustomobject]@{ field = $fldName; factors = $fact; hashes = $hashes; tmpCount = $hashes.Count }
}

$failures = 0
foreach ($c in $cases) {
    Write-Host ""
    Write-Host ("=== {0} ===" -f $c.name)
    $nShow = $c.N
    if ($nShow.Length -gt 24) { $nShow = $nShow.Substring(0, 21) + "..." }
    Write-Host ("  N={0}  B1={1}  sigma={2}  curves={3}" -f $nShow, $c.B1, $c.sigma, $Curves)

    $res = @{}
    foreach ($b in $c.backends) {
        $tag = ""; $field = ""; $backendName = "simd"
        switch ($b) {
            "gmp"       { $tag = "gmp";       $field = "off";  $backendName = "gmp" }
            "simd-mont" { $tag = "simd/mont"; $field = "off";  $backendName = "simd" }
            "simd-mers" { $tag = "simd/fold"; $field = "on";   $backendName = "simd" }
            "simd-auto" { $tag = "simd/auto"; $field = "auto"; $backendName = "simd" }
            default     { throw "unknown backend tag $b" }
        }
        $dir = Join-Path $work ("{0}_{1}" -f ($c.name -replace '[^\w]', ''), ($b -replace '-', ''))
        $res[$b] = Run-Backend -backend $backendName -field $field -N $c.N -B1 $c.B1 -sigma $c.sigma -dir $dir
        Write-Host ("  [{0,-10}] field={1,-11} hits={2}/{3}  tmp={4}" -f `
                    $tag, $res[$b].field, $res[$b].factors.Count, $Curves, $res[$b].tmpCount)
    }

    # 1) same hit set and same factor values
    $ref = $c.backends[0]
    foreach ($b in $c.backends) {
        if ($b -eq $ref) { continue }
        $a = $res[$ref].factors; $z = $res[$b].factors
        $same = ($a.Count -eq $z.Count)
        if ($same) {
            foreach ($k in $a.Keys) {
                if (-not $z.ContainsKey($k) -or $z[$k] -ne $a[$k]) { $same = $false; break }
            }
        }
        if (-not $same) {
            Write-Host ("  FAIL: hit set / factor values differ: {0} vs {1}" -f $ref, $b)
            $failures++
        }
    }
    # 2) .tmp archives byte-identical
    foreach ($b in $c.backends) {
        if ($b -eq $ref) { continue }
        $bad = @()
        foreach ($k in $res[$ref].hashes.Keys) {
            if (-not $res[$b].hashes.ContainsKey($k)) { $bad += "$k (missing)"; continue }
            if ($res[$b].hashes[$k] -ne $res[$ref].hashes[$k]) { $bad += $k }
        }
        if ($res[$b].tmpCount -ne $res[$ref].tmpCount) {
            $bad += ("tmp count {0} != {1}" -f $res[$b].tmpCount, $res[$ref].tmpCount)
        }
        if ($bad.Count -gt 0) {
            Write-Host ("  FAIL: archives differ: {0} vs {1}: {2}" -f $ref, $b, ($bad -join ", "))
            $failures++
        } else {
            Write-Host ("  ok  : {0} == {1}  ({2} .tmp byte-identical)" -f $ref, $b, $res[$ref].tmpCount)
        }
    }
    # 3) the fold domain must be picked exactly when N = 2^k-1
    if ($res.ContainsKey("simd-auto")) {
        $isMers = ($c.N -match '^\(2\^\d+-1\)$')
        if (-not $isMers -and $res["simd-auto"].field -ne "montgomery") {
            Write-Host ("  FAIL: non-Mersenne N selected field {0}" -f $res["simd-auto"].field); $failures++
        }
        if ($isMers -and $res["simd-auto"].field -ne "mersenne") {
            Write-Host ("  FAIL: Mersenne N did not select the fold domain (field={0})" -f $res["simd-auto"].field); $failures++
        }
    }
}

Write-Host ""
if ($failures -eq 0) {
    Write-Host ("RESULT: PASS  ({0} cases; three backends agree on hits and archives)" -f $cases.Count)
    exit 0
}
Write-Host ("RESULT: FAIL  ({0} mismatches)" -f $failures)
exit 1
