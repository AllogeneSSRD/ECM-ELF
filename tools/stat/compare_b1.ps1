# cmp_b1.ps1 -- for a sweep of B1 on M3001 / sigma=20260922, compare the scalar (gmp)
# and SIMD (fold / montgomery) backends on:
#   * how many of 8 curves report a factor
#   * the saved final point (Qx,Qz of curve 0) -- identical bytes => identical ladder
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$root = $repo
$scratch = Join-Path $PSScriptRoot "_run"
New-Item -ItemType Directory -Force -Path $scratch | Out-Null
$exe = Join-Path $root "build_vs18\Release\ecm.exe"
$dump = Join-Path $repo "build_vs18\tools\dump_tmp.exe"
$env:PATH = (Join-Path $root "third_party\gmp-zen3\dist\bin") + ";" + $env:PATH
$N = "(2^3001-1)"
$B1s = @(100, 500, 1000, 2000, 5000, 10000, 20000)

function Run-One([string]$backend, [string]$mers, [int]$B1) {
    $d = Join-Path $scratch ("sw_{0}_{1}_{2}" -f $backend, $mers, $B1)
    Remove-Item $d -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    $argv = @("--edwards", "--edwards-backend", $backend, "--edwards-threads", "1",
              "--edwards-naf-w", "12", "--edwards-mersenne", $mers,
              "-gpucurves", "8", "-sigma", "20260922", "--tmp-dir", $d, "$B1", "0")
    $out = $N | & $exe @argv 2>&1
    $hits = @($out | Select-String -Pattern "factor found").Count
    $tmp = Join-Path $d ("e0003001_B{0}_c000001.tmp" -f $B1)
    $qz = "-"
    if (Test-Path $tmp) {
        $dl = & $dump $tmp 2>&1
        $line = ($dl | Select-String -Pattern "^  Qz=" | Select-Object -First 1)
        if ($line) { $qz = ($line.Line -replace '^\s*Qz=', '').Trim() }
    }
    return [pscustomobject]@{ backend = $backend; mers = $mers; B1 = $B1; hits = $hits; Qz = $qz }
}

foreach ($b1 in $B1s) {
    $g = Run-One "gmp" "auto" $b1
    $m = Run-One "simd" "on" $b1
    $x = Run-One "simd" "off" $b1
    $same_gm = ($g.Qz -eq $m.Qz)
    $same_mx = ($m.Qz -eq $x.Qz)
    $qlen = if ($g.Qz -eq "-") { 0 } else { $g.Qz.Length }
    Write-Host ("B1={0,-6} hits: gmp={1} simd-mers={2} simd-mont={3} | curve0 Qz: gmp==mers? {4}  mers==mont? {5}  (gmp len {6})" -f `
                $b1, $g.hits, $m.hits, $x.hits, $same_gm, $same_mx, $qlen)
}
