# A/B: stock vcpkg GMP vs zen3-kernel GMP on the Edwards stage-1 workload.
# Builds the same benchmark source twice (identical MSVC flags), one linked
# against each GMP, and runs them alternately to average out thermal drift.
$ErrorActionPreference = "Continue"
$root = "D:\code\MPA-OpenCl"
$gmpBase = "D:\code\vcpkg\installed\x64-windows"
$gmpZen = "$root\third_party\gmp-zen3\dist"
$ab = "$root\tools\bench\_abtest"
Remove-Item -Recurse -Force $ab -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path "$ab\base", "$ab\zen3" | Out-Null

$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
$src = "$root\tools\bench\ecm_edwards_speed.cpp"
$cpu = "$root\src\cpu\ecm_edwards_cpu.cpp"

function Build($inc, $lib, $exe) {
    $line = "cl /nologo /O2 /EHsc /utf-8 /I `"$inc`" /I `"$root\src\cpu`" `"$src`" `"$cpu`" /Fe:`"$exe`" /link `"$lib`""
    $bat = "@echo off`r`ncall `"$vcvars`" >nul`r`n$line`r`n"
    Set-Content -Path "$env:TEMP\ab_build.bat" -Value $bat -Encoding ASCII
    cmd /c "$env:TEMP\ab_build.bat" | Out-Null
    return ($LASTEXITCODE -eq 0)
}

Write-Host "building base..."
$ok1 = Build "$gmpBase\include" "$gmpBase\lib\gmp.lib" "$ab\base\speed.exe"
Write-Host "building zen3..."
$ok2 = Build "$gmpZen\include" "$gmpZen\lib\gmp.dll.lib" "$ab\zen3\speed.exe"
if (-not $ok1 -or -not $ok2) { Write-Host "BUILD FAILED base=$ok1 zen3=$ok2"; exit 1 }
Copy-Item "$gmpBase\bin\gmp-10.dll" "$ab\base\" -Force
Copy-Item "$gmpZen\bin\gmp-10.dll"   "$ab\zen3\" -Force

Write-Host ""
Write-Host "=== A/B Edwards stage-1 (M991 B1=1e6 w=8) : alternating rounds ==="
$res = @{ base = @(); zen3 = @() }
foreach ($round in 1..3) {
    foreach ($tag in @("base","zen3")) {
        $out = & "$ab\$tag\speed.exe" 991 1000000 8 2 2>&1
        $line = ($out | Select-String -Pattern "^min=").Line
        if ($line -match "min=([0-9.]+)") { $res[$tag] += [double]$Matches[1] }
        Write-Host ("  round {0} {1,-5}: {2}" -f $round, $tag, $line)
    }
}
Write-Host ""
foreach ($tag in @("base","zen3")) {
    $v = $res[$tag] | Sort-Object
    Write-Host ("{0,-5}: min={1:N3}s median={2:N3}s  runs=[{3}]" -f $tag, $v[0], $v[[int]($v.Count/2)], (($v | ForEach-Object { "{0:N3}" -f $_ }) -join ", "))
}
$b = ($res["base"] | Sort-Object)[0]
$z = ($res["zen3"] | Sort-Object)[0]
Write-Host ("SPEEDUP (min/min): {0:N3}x" -f ($b / $z))
