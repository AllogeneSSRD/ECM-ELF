# Edward stage-1 multi-curve threading scaling benchmark.
param(
    [int]$Curves = 24,
    [long]$B1 = 1000000,
    [int]$Bits = 991,
    [uint64]$Sigma = 105413044550089,
    [string]$ThreadList = "1,4,8,12,24"
)

$Threads = $ThreadList.Split(",") | ForEach-Object { [int]$_.Trim() }

$env:PATH = "D:\code\vcpkg\installed\x64-windows\bin;" + $env:PATH
$exe = "D:\code\MPA-OpenCl\build_vs18\Release\ecm.exe"
$n = "2^$Bits-1"

Write-Host "=== Edwards stage-1 thread scaling: curves=$Curves bits=$Bits B1=$B1 ==="
Write-Host "threads  wall_s   curve_s  speedup  factors"
$base = 0.0
foreach ($t in $Threads) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $out = $n | & $exe --edwards -sigma $Sigma -gpucurves $Curves --edwards-threads $t $B1 2>&1
    $sw.Stop()
    $wall = $sw.Elapsed.TotalSeconds
    if ($base -eq 0.0) { $base = $wall }
    $nf = ($out | Select-String -Pattern 'factor\[' ).Count
    $line = "$t`t$([math]::Round($wall,2))`t$([math]::Round($wall/$Curves,2))`t$([math]::Round($base/$wall,2))`t$nf"
    Write-Host $line
}
Write-Host "done"
