# ECM Stage1 Factor Validation Test
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/test_validate_factors.ps1

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Resolve-Path "$ScriptDir\"
$EcmBin = Join-Path $RootDir "build_rel\Release\ecm.exe"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ECM Factor Validation Test" -ForegroundColor Cyan
Write-Host "  Binary: $EcmBin" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$Tests = @(
    @{ L="2^151-1";  N="(2^151-1)";  S="3:2026";       C=1;  B="1e4";  E="391612124215324515959" },
    @{ L="2^347-1";  N="(2^347-1)";  S="3:561219477";  C=1;  B="1e4";  E="14143189112952632419639" },
    @{ L="2^421-1";  N="(2^421-1)";  S="3:268526266";  C=1;  B="1e4";  E="614002928307599" },
    @{ L="2^677-1";  N="(2^677-1)";  S="3:4001686290"; C=1;  B="1e3";  E="1943118631" },
    @{ L="2^991-1";  N="(2^991-1)";  S="3:822692423";  C=1;  B="1e3";  E="231620367206687" },
    @{ L="2^2017-1";  N="(2^2017-1)";  S="3:423561925";  C=1;  B="1e3";  E="9338711" },
    @{ L="2^3041-1";  N="(2^3041-1)";  S="3:122189128";  C=1;  B="1e2";  E="135391639199" }
)

$Pass = 0; $Fail = 0; $Res = @()

foreach ($T in $Tests) {
    $Idx = [array]::IndexOf($Tests, $T)
    Write-Host ("[{0}] sigma={1} B1={2} ..." -f $T.L, $T.S, $T.B) -NoNewline

    # Write N expression to a temp stdin file (no escaping needed for file)
    $StdinFile = Join-Path $env:TEMP "ecm_in_${Idx}.txt"
    $OutFile = Join-Path $env:TEMP "ecm_out_${Idx}.txt"
    Set-Content -Path $StdinFile -Value $T.N -Encoding ASCII -NoNewline

    # Bat: feed stdin from file, redirect stdout+stderr to output file
    $BatContent = "@echo off`r`n`"$EcmBin`" -v -d 1 --curve mont --target gpu -sigma $($T.S) -gpucurves $($T.C) $($T.B) 0 <`"$StdinFile`" >`"$OutFile`" 2>&1"
    $BatFile = Join-Path $env:TEMP "ecm_run_${Idx}.bat"
    Set-Content -Path $BatFile -Value $BatContent -Encoding ASCII

    try {
        $null = cmd /c $BatFile 2>$null
        if (Test-Path $OutFile) {
            $OutStr = Get-Content -Path $OutFile -Raw -ErrorAction SilentlyContinue
        } else {
            $OutStr = ""
        }
    } catch {
        $OutStr = ""
    }
    Remove-Item $BatFile, $StdinFile, $OutFile -ErrorAction SilentlyContinue -Force

    $Found = $null
    $OpsLine = ""
    if ($OutStr -match 'GPU: stage1 operators:\s*([^\r\n]+)') {
        $OpsLine = $Matches[1]
    }
    if ($OutStr -match 'factor\[\d+\]\s*=\s*(\d+)') {
        $Found = $Matches[1]
    }

    if ($Found -eq $T.E) {
        Write-Host " PASS" -ForegroundColor Green
        $Pass++
        $Res += @{ L=$T.L; S="PASS"; E=$T.E; F=$Found }
    } elseif ($Found) {
        Write-Host (" MISMATCH: {0}" -f $Found) -ForegroundColor Red
        $Fail++
        $Res += @{ L=$T.L; S="FAIL"; E=$T.E; F=$Found }
    } else {
        Write-Host " NOFACTOR" -ForegroundColor Red
        $Fail++
        $Res += @{ L=$T.L; S="FAIL"; E=$T.E; F="(none)" }
    }
    if ($OpsLine) {
        Write-Host ("       operators: {0}" -f $OpsLine) -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ("  Results: {0} PASS, {1} FAIL, {2} TOTAL" -f $Pass, $Fail, $Tests.Count) -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

foreach ($R in $Res) {
    if ($R.S -eq "PASS") {
        Write-Host ("  [PASS] {0}: {1}" -f $R.L, $R.F) -ForegroundColor Green
    } else {
        Write-Host ("  [FAIL] {0}: got={1} expected={2}" -f $R.L, $R.F, $R.E) -ForegroundColor Red
    }
}

exit $Fail
