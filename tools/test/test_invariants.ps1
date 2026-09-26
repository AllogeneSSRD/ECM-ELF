$ErrorActionPreference = "Continue"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$root = $repo
$scratch = Join-Path $PSScriptRoot "_run"
New-Item -ItemType Directory -Force -Path $scratch | Out-Null
$exe  = Join-Path $root "build_vs18\Release\ecm.exe"
$env:PATH = (Join-Path $root "third_party\gmp-zen3\dist\bin") + ";" + $env:PATH

$cases = @(
    @("M677",  "677",  "6581585141005897",  "1943118631"),
    @("M991",  "991",  "105413044550089",   "8218291649"),
    @("M4003", "4003", "2027329164697536",  "16756559")
)

foreach ($field in @("on","off")) {
    foreach ($c in $cases) {
        $tag=$c[0]; $n=$c[1]; $sigma=$c[2]; $expect=$c[3]
        $t = Join-Path $scratch ("inv_$tag`_$field")
        if (Test-Path $t) { Remove-Item -Recurse -Force $t }
        New-Item -ItemType Directory -Force -Path $t | Out-Null
        $a = @("--edwards","--edwards-backend","simd","--edwards-threads","1","--edwards-naf-w","12",
               "--edwards-mersenne",$field,"-gpucurves","2","-sigma",$sigma,
               "--tmp-dir",$t,"1000000","0")
        $sw=[System.Diagnostics.Stopwatch]::StartNew()
        "(2^$n-1)" | & $exe @a *>&1 | Out-File -FilePath (Join-Path $t "out.log") -Encoding ASCII
        $sw.Stop()
        $fac = (Get-Content (Join-Path $t "out.log") | Select-String "factor\[0\]=" | ForEach-Object { ($_.Line -split '=')[-1].Trim() }) -join ""
        $fld = (Get-Content (Join-Path $t "out.log") | Select-String "field           :" | ForEach-Object { ($_.Line -split ':')[-1].Trim() })
        $ok = if ($fac -eq $expect) { "OK" } else { "MISMATCH(expected $expect)" }
        Write-Host ("{0,-6} {1,-4} wall={2,7:N1}s factor={3,-22} {4}" -f $tag, $field, $sw.Elapsed.TotalSeconds, $fac, $ok)
    }
}
