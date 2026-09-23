$ErrorActionPreference = "Stop"

$root = "D:\code\MPA-OpenCl"
Set-Location $root

$ecm = ".\build\Debug\ecm.exe"
$addsub = ".\build\Debug\opencl_ecm_addsub.exe"
$outDir = Join-Path $root "bench"
if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$ecmCsv = Join-Path $outDir "wg_stage1_scaling.csv"
$opCsv = Join-Path $outDir "wg_operator_scaling.csv"

"bits,input_expr,mode,gputime_ms,factor_found" | Out-File -FilePath $ecmCsv -Encoding ascii
"bits,mode,kernel,ms,ops_per_s,private_mem_bytes,local_mem_bytes,preferred_wg_multiple,max_wg_size" | Out-File -FilePath $opCsv -Encoding ascii

$cases = @(
    @{ bits = 1024; expr = "(2^1021-1)" },
    @{ bits = 2048; expr = "(2^2039-1)" },
    @{ bits = 4096; expr = "(2^4093-1)" },
    @{ bits = 8192; expr = "(2^8191-1)" }
)

function Run-EcmCase($bits, $expr, $modeName, $disableWg) {
    if ($disableWg) { $env:ECM_DISABLE_MONT_WG = "1" } else { Remove-Item Env:ECM_DISABLE_MONT_WG -ErrorAction SilentlyContinue }
    $env:ECM_PROFILE_OPS = "1"
    $env:ECM_GPU_DUMP = "0"
    $cmd = "Set-Location '$root'; '$expr' | $ecm -v -gpu -sigma 3:12345678 -gpucurves 32 2000 0"
    $out = powershell -NoProfile -Command $cmd
    $gputime = ""
    $found = "0"
    foreach ($line in $out) {
        if ($line -match "opencl_ecm_stage1 returned:\s+([0-9\-]+)\s+gputime=([0-9\.]+)\s+ms") {
            $gputime = $Matches[2]
        }
        if ($line -match "factor found in Step 1") {
            $found = "1"
        }
    }
    if ($gputime -eq "") { $gputime = "-1" }
    "$bits,$expr,$modeName,$gputime,$found" | Out-File -FilePath $ecmCsv -Append -Encoding ascii
}

function Run-OpCase($bits, $modeName, $useWg) {
    $tmp = Join-Path $outDir ("tmp_ops_{0}_{1}.csv" -f $modeName, $bits)
    $env:ECM_BENCH_CSV = $tmp
    $wgArg = if ($useWg) { "--use-wg --tpi 8" } else { "--no-wg --tpi 8" }
    $cmd = "Set-Location '$root'; $addsub --bits $bits $wgArg 400 192 10"
    powershell -NoProfile -Command $cmd | Out-Null
    if (Test-Path $tmp) {
        $lines = Get-Content $tmp
        foreach ($line in $lines) {
            if ($line -match "^kernel,") { continue }
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            "$bits,$modeName,$line" | Out-File -FilePath $opCsv -Append -Encoding ascii
        }
        Remove-Item $tmp -ErrorAction SilentlyContinue
    }
}

foreach ($c in $cases) {
    Run-EcmCase -bits $c.bits -expr $c.expr -modeName "wg" -disableWg $false
    Run-EcmCase -bits $c.bits -expr $c.expr -modeName "priv" -disableWg $true
    Run-OpCase -bits $c.bits -modeName "wg" -useWg $true
    Run-OpCase -bits $c.bits -modeName "priv" -useWg $false
}

Remove-Item Env:ECM_DISABLE_MONT_WG -ErrorAction SilentlyContinue
Remove-Item Env:ECM_BENCH_CSV -ErrorAction SilentlyContinue
Remove-Item Env:ECM_PROFILE_OPS -ErrorAction SilentlyContinue
Remove-Item Env:ECM_GPU_DUMP -ErrorAction SilentlyContinue

Write-Output "Wrote: $ecmCsv"
Write-Output "Wrote: $opCsv"
