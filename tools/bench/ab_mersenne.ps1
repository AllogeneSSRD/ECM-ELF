$ErrorActionPreference = "Continue"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$root = $repo
$scratch = Join-Path $PSScriptRoot "_run"
New-Item -ItemType Directory -Force -Path $scratch | Out-Null
$exe  = Join-Path $root "build_vs18\Release\ecm.exe"
$env:PATH = (Join-Path $root "third_party\gmp-zen3\dist\bin") + ";" + $env:PATH
$N = "(2^3001-1)"

function Run-One([string]$field, [string]$tmpdir, [int]$curves, [string]$B1, [string]$sigma) {
    if (Test-Path $tmpdir) { Remove-Item -Recurse -Force $tmpdir }
    New-Item -ItemType Directory -Force -Path $tmpdir | Out-Null
    $log = Join-Path $tmpdir "out.log"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $args = @("--edwards", "--edwards-backend", "simd", "--edwards-threads", "1",
              "--edwards-naf-w", "12", "--edwards-mersenne", $field,
              "-gpucurves", "$curves", "--tmp-dir", $tmpdir, $B1, "0")
    if ($sigma) { $args += @("-sigma", $sigma) }
    $N | & $exe @args *>&1 | Out-File -FilePath $log -Encoding ASCII
    $sw.Stop()
    $field_line = (Get-Content $log | Select-String "field           :" | Select-Object -First 1)
    Write-Host ("  [{0,-4}] wall={1,7:N2}s  {2}" -f $field, $sw.Elapsed.TotalSeconds, ($field_line -replace '.*field\s+: ', ''))
    return $sw.Elapsed.TotalSeconds
}

Write-Host "=== byte identity: same sigma, B1=1e5, mersenne vs montgomery ==="
$sa = Join-Path $scratch "idm"
$sb = Join-Path $scratch "idt"
$t1 = Run-One "on"  $sa 8 "100000" "20260922"
$t2 = Run-One "off" $sb 8 "100000" "20260922"
$ha = Get-ChildItem $sa -Filter *.tmp | Sort-Object Name | ForEach-Object { (Get-FileHash $_.FullName -Algorithm SHA256).Hash }
$hb = Get-ChildItem $sb -Filter *.tmp | Sort-Object Name | ForEach-Object { (Get-FileHash $_.FullName -Algorithm SHA256).Hash }
Write-Host ("  .tmp count: {0} vs {1}" -f $ha.Count, $hb.Count)
$same = ($ha.Count -eq $hb.Count) -and ($ha.Count -gt 0)
for ($i = 0; $i -lt [Math]::Min($ha.Count, $hb.Count); $i++) {
    if ($ha[$i] -ne $hb[$i]) { $same = $false; Write-Host "  MISMATCH at file $i" }
}
Write-Host ("  byte-identical: {0}" -f $same)

Write-Host ""
Write-Host "=== end-to-end A/B: 8 curves, M3001, B1=1e6, alternating min-of-3 ==="
$best = @{ on = 1e30; off = 1e30 }
for ($round = 1; $round -le 3; $round++) {
    foreach ($f in @("on", "off")) {
        $t = Run-One $f (Join-Path $scratch "ab_$f") 8 "1000000" ""
        if ($t -lt $best[$f]) { $best[$f] = $t }
    }
}
Write-Host ""
Write-Host ("  mersenne   : {0:N2}s  = {1:N2} s/curve" -f $best["on"], ($best["on"] / 8))
Write-Host ("  montgomery : {0:N2}s  = {1:N2} s/curve" -f $best["off"], ($best["off"] / 8))
Write-Host ("  speedup    : {0:N2}x" -f ($best["off"] / $best["on"]))
