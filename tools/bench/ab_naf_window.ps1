$ErrorActionPreference = "Continue"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$root = $repo
$scratch = Join-Path $PSScriptRoot "_run"
New-Item -ItemType Directory -Force -Path $scratch | Out-Null
$exe  = Join-Path $root "build_vs18\Release\ecm.exe"
$env:PATH = (Join-Path $root "third_party\gmp-zen3\dist\bin") + ";" + $env:PATH
$N = "(2^3001-1)"

function Run-W([int]$w) {
    $t = Join-Path $scratch "wt_$w"
    if (Test-Path $t) { Remove-Item -Recurse -Force $t }
    New-Item -ItemType Directory -Force -Path $t | Out-Null
    $a = @("--edwards","--edwards-backend","simd","--edwards-threads","1","--edwards-naf-w","$w",
           "--edwards-mersenne","on","-gpucurves","8","--tmp-dir",$t,"1000000","0")
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $N | & $exe @a *>&1 | Out-File -FilePath (Join-Path $t "out.log") -Encoding ASCII
    $sw.Stop()
    return $sw.Elapsed.TotalSeconds
}

$best = @{}
foreach ($round in 1..3) {
    foreach ($w in 12, 8, 10) {
        $t = Run-W $w
        if (-not $best.ContainsKey($w) -or $t -lt $best[$w]) { $best[$w] = $t }
        Write-Host ("  round {0} w={1,-3} {2,7:N2}s" -f $round, $w, $t)
    }
}
Write-Host ""
foreach ($w in $best.Keys | Sort-Object) {
    Write-Host ("w={0,-3} best {1,7:N2}s = {2:N2} s/curve" -f $w, $best[$w], ($best[$w] / 8))
}
