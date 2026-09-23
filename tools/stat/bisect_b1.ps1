# bisect_b1.ps1 -- on the known-failing case (M3001, sigma=20260922) find the smallest
# B1 where the scalar (gmp) backend and the SIMD backends disagree about the factor.
# All runs use the same sigma set, so each B1 is a paired comparison.
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$root = $repo
$scratch = Join-Path $PSScriptRoot "_run"
New-Item -ItemType Directory -Force -Path $scratch | Out-Null
$exe = Join-Path $root "build_vs18\Release\ecm.exe"
$env:PATH = (Join-Path $root "third_party\gmp-zen3\dist\bin") + ";" + $env:PATH
$N = "(2^3001-1)"
$B1s = @(10000, 20000, 30000, 50000, 75000, 100000)

function Run-One([string]$backend, [string]$mers, [int]$B1) {
    $d = Join-Path $scratch "bisect_$backend$mers$B1"
    Remove-Item $d -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    $argv = @("--edwards", "--edwards-backend", $backend, "--edwards-threads", "1",
              "--edwards-naf-w", "12", "--edwards-mersenne", $mers,
              "-gpucurves", "8", "-sigma", "20260922", "--tmp-dir", $d, "$B1", "0")
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $out = $N | & $exe @argv 2>&1
    $sw.Stop()
    $found = @($out | Select-String -Pattern "factor found").Count
    $f0 = ($out | Select-String -Pattern "factor\[0\]=" | Select-Object -First 1)
    $f0v = if ($f0) { ($f0.Line -split "=")[-1].Trim() } else { "-" }
    return [pscustomobject]@{ backend = $backend; mers = $mers; B1 = $B1;
                              curves_hit = $found; factor0 = $f0v;
                              secs = [math]::Round($sw.Elapsed.TotalSeconds, 1) }
}

$rows = @()
foreach ($b1 in $B1s) {
    $rows += Run-One "gmp" "auto" $b1
    $rows += Run-One "simd" "on" $b1
    $rows += Run-One "simd" "off" $b1
}
$rows | Format-Table -AutoSize | Out-String -Width 160
