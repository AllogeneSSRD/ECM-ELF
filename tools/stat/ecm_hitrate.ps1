# ---------------------------------------------------------------------------
# ecm_hitrate.ps1 -- stage-1 hit-rate statistics: compare the two backends
# (scalar gmp vs SIMD, both domains) against an independent reference.
#
# Small primes come from tools/ecm_prob/data/primes/bits<b>.bin (primesieve,
# uint64 little-endian).  Each prime is embedded in a large composite:
#     N = p * (2^521-1)
# This is REQUIRED: with N = p (prime) a stage-1 hit yields gcd == N and the
# driver discards it as a trivial factor, so the measured rate would be 0 and
# prove nothing.  With the cofactor, a hit is reported as the proper factor p.
#
# Reference values (independent Python implementation in tools/ecm_prob):
#   Edwards Z/2xZ/8, B1=256, 1 curve/prime, all 38635 20-bit primes
#     -> 12620/38635 = 32.66%   (out/measure_20_256.json)
#   measured here with 1000 sampled primes x 8 curves:
#     simd/auto = simd/mont = gmp = 2611/8000 = 32.6375%   (identical hit sets)
#   B1=1e5 (long ladders), 60 primes x 8 curves:
#     all three backends = 477/480 = 99.375%               (identical hit sets)
#
# usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/stat/ecm_hitrate.ps1
#   ... -File tools/stat/ecm_hitrate.ps1 -Count 200 -B1 1000 -Bits 24
#   ... -File tools/stat/ecm_hitrate.ps1 -Only gmp          # single backend
# ---------------------------------------------------------------------------
param(
    [int]$Count = 1000,
    [int]$B1 = 256,
    [int]$Curves = 8,
    [int]$Bits = 20,
    [string]$Only = ""
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$root = $repo
$scratch = Join-Path $PSScriptRoot "_run"
New-Item -ItemType Directory -Force -Path $scratch | Out-Null

$exe = Join-Path $repo "build_vs18\Release\ecm.exe"
if (-not (Test-Path $exe)) { Write-Host "FAIL: ecm.exe not found: $exe"; exit 2 }
$env:PATH = (Join-Path $repo "third_party\gmp-zen3\dist\bin") + ";" + $env:PATH

$bin = Join-Path $repo ("tools\ecm_prob\data\primes\bits{0}.bin" -f $Bits)
if (-not (Test-Path $bin)) { Write-Host "FAIL: prime data not found: $bin"; exit 2 }
$bytes = [System.IO.File]::ReadAllBytes($bin)
$total = [int]($bytes.Length / 8)
$stride = [math]::Max(1, [math]::Floor($total / $Count))
$q = [System.Numerics.BigInteger]::Pow(2, 521) - 1

$backends = @("simd-auto", "simd-mont", "gmp")
if ($Only) { $backends = @($Only -split ",") }

$primes = @()
for ($i = 0; $i -lt $Count; $i++) {
    $idx = $i * $stride
    if ($idx * 8 + 8 -gt $bytes.Length) { break }
    $primes += [System.BitConverter]::ToUInt64($bytes, $idx * 8)
}

Write-Host ("primes : {0} sampled from {1} ({2}-bit, stride {3})" -f $primes.Count, $total, $Bits, $stride)
Write-Host ("config : B1={0}, curves/prime={1}, N = p * (2^521-1)" -f $B1, $Curves)
Write-Host ""

$summary = @()
foreach ($be in $backends) {
    $backend = "simd"; $mersenne = "auto"
    switch ($be) {
        "simd-auto" { $backend = "simd"; $mersenne = "auto" }
        "simd-mers" { $backend = "simd"; $mersenne = "on"   }
        "simd-mont" { $backend = "simd"; $mersenne = "off"  }
        "gmp"       { $backend = "gmp";  $mersenne = "auto" }
        default     { throw "unknown backend $be" }
    }
    $tmp = Join-Path $scratch "tmp_$be"
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null

    $hits = 0; $nCurves = 0; $badFactor = 0; $example = @()
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    for ($i = 0; $i -lt $primes.Count; $i++) {
        $p = $primes[$i]
        $N = ([System.Numerics.BigInteger]$p) * $q
        $sigma = 1000003 + 7919 * $i
        Get-ChildItem $tmp -File -ErrorAction SilentlyContinue | Remove-Item -Force
        $argv = @("--edwards", "--edwards-backend", $backend, "--edwards-threads", "1",
                  "--edwards-naf-w", "12", "--edwards-mersenne", $mersenne,
                  "-gpucurves", [string]$Curves, "-sigma", [string]$sigma,
                  "--tmp-dir", $tmp, [string]$B1, "0")
        $out = "$N" | & $exe @argv 2>&1
        $nCurves += $Curves
        $hits += @($out | Select-String -Pattern "factor found").Count
        foreach ($line in @($out | Select-String -Pattern "factor\[\d+\]=")) {
            $v = [System.Numerics.BigInteger]::Parse(($line.Line -split "=")[-1].Trim())
            if ($v -ne [System.Numerics.BigInteger]$p) {
                $badFactor++
                if ($example.Count -lt 3) { $example += ("p=$p got=$v") }
            } elseif ($example.Count -lt 3) { $example += ("p=$p -> $v") }
        }
    }
    $sw.Stop()
    $rate = 100.0 * $hits / [math]::Max(1, $nCurves)
    Write-Host ("[{0,-10}] hits={1,6} / {2,6} curves = {3,6:N3}%   ({4:N1}s, {5:N4} s/curve)" -f `
                $be, $hits, $nCurves, $rate, $sw.Elapsed.TotalSeconds,
                ($sw.Elapsed.TotalSeconds / [math]::Max(1, $nCurves)))
    if ($badFactor -gt 0) {
        Write-Host ("             !! {0} hits reported a factor != p : {1}" -f $badFactor, ($example -join "; "))
    } elseif ($example.Count -gt 0) {
        Write-Host ("             e.g. {0}" -f ($example -join "; "))
    }
    $summary += [pscustomobject]@{ backend = $be; hits = $hits; curves = $nCurves; pct = $rate }
}

Write-Host ""
Write-Host "=== summary (reference: Edwards Z/2xZ/8 B1=256 -> 32.66% over all 38635 primes) ==="
$summary | Format-Table -AutoSize
