#Requires -Version 5.1
<#
.SYNOPSIS
    D3 acceptance: the hit line carries factor + curve + sigma + param + method + save
    (docs/DEV_ECM_GUI.md section 11, D3).

.DESCRIPTION
    Runs the real driver in QUEUE mode (the mode the GUI uses) on M677 with B1=1e6 and
    8 curves, in a sandbox directory, and checks every hit line it prints:

        factor[i]=<decimal> curve=<i> sigma=<64-bit> param=<p> method=<m> save=<name>

    M677 is used because its small factor 1943118631 (31 bits) is found by most curves at
    B1=1e6 (measured 6..11 of 16 curves per run), so this test does not depend on luck --
    unlike M991 at B1=1e4, which hit 3/8 curves once and 0/128 the next time.

    Asserted:
      * at least one hit, and every hit line has the six D3 fields;
      * curve == the index inside factor[i];
      * sigma - curve is the same constant for all hits of one batch (so the reported
        sigma really is from the sequence the driver used);
      * param/method/save match the task;
      * the factors really divide 2^677-1, and the known M677 factor shows up.

    NOTE (unrelated finding, measured while writing this test): single-run CLI GPU mode
    ("echo N | ecm_cuda ... -gpu ...") dies inside CGBN with "invalid modulus (it must be
    odd)" for every parametrization, while the identical task through the queue manager
    works. The D3 printing cannot be the cause -- both modes fill the same per-curve sigma
    array in the same function, and the queue path passes -- so this is a pre-existing
    CLI-only defect (docs/DEV_ECM_GUI.md, TODO).

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_hit_fields.ps1
#>
param(
    [string]$EcmCuda = "",
    [string]$Sandbox = "",
    [int]$Exp = 677,
    [string]$B1 = "1e6",
    [int]$Curves = 16,          # 16 curves make a hit very likely; the retry loop below makes it certain enough
    [int]$Device = 0,
    [int]$MaxAttempts = 3       # M677/B1=1e6 is probabilistic: retry with a different sigma
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $EcmCuda) {
    foreach ($cand in @("$repoRoot\build_cuda_cmake\ecm_cuda.exe", "$repoRoot\build_vs18\Release\ecm_cuda.exe")) {
        if (Test-Path $cand) { $EcmCuda = $cand; break }
    }
}
if (-not (Test-Path $EcmCuda)) { Write-Host "FAIL: ecm_cuda.exe not found" -ForegroundColor Red; exit 2 }
if (-not $Sandbox) { $Sandbox = Join-Path $PSScriptRoot '_run\hit_fields' }
if (Test-Path $Sandbox) { Remove-Item $Sandbox -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $Sandbox 'saves') | Out-Null

$script:pass = 0
$script:fail = 0
function Check([string]$name, [bool]$ok, [string]$detail = "") {
    if ($ok) { $script:pass++; Write-Host ("  [ok]   " + $name) }
    else {
        $script:fail++
        Write-Host ("  [FAIL] " + $name + $(if ($detail) { " -- " + $detail } else { "" })) -ForegroundColor Red
    }
}

$enc = New-Object System.Text.UTF8Encoding($false)
$saveName = "m${Exp}_1e6.save"
$task = 'ECMSTAGE2=1,2,{0},-1,"{1}",0,0,{2}' -f $Exp, $saveName, $Curves
$todo = Join-Path $Sandbox 'worktodo.txt'
$ini = Join-Path $Sandbox 'ecm.ini'
$finished = Join-Path $Sandbox 'finished.txt'
$saves = Join-Path $Sandbox 'saves'
$logFile = Join-Path $Sandbox 'screen_1.log'

[System.IO.File]::WriteAllText($todo, ("[Worker #1]`r`n" + $task + "`r`n"), $enc)
# NOTE: parenthesise every concatenation inside an array literal -- the comma binds
# tighter than '+', so "key = " + $v would silently become two ini lines.
$iniLines = @(
    'method = gpu',
    ('gpucurves = ' + $Curves),
    ('tmp_dir = ' + $saves),
    ('finished = ' + $finished),
    ('worktodo = ' + $todo),
    'verbose = 0',
    'ckpt_seconds = 0',
    '',
    '[Worker #1]',
    ('device = ' + $Device),
    ('log_file = ' + $logFile)
)
[System.IO.File]::WriteAllText($ini, (($iniLines -join "`r`n") + "`r`n"), $enc)

Write-Host ("driver  : " + $EcmCuda)
Write-Host ("task    : {0}  (2^{1}-1, B1={2}, {3} curves, device {4})" -f $task, $Exp, $B1, $Curves, $Device)
Write-Host ""

Write-Host "[1] run the queue manager on the task"
# A hit is probabilistic (measured 6..11 of 16 curves), so a single run can legitimately report
# zero and the old version of this test then failed for luck, not for a defect (2026-09-29). The
# queue mode takes the curve sigma from the ini's `sigma =`, so each attempt uses a different
# FIXED sigma -- reproducible, and a retry is a new sample instead of the same dice again.
$cmd = '"{0}" -ini "{1}" --worker 1 < NUL 2>&1' -f $EcmCuda, $ini
$out = ''
$hits = @()
$done = @()
$attempt = 0
while ($attempt -lt $MaxAttempts -and $hits.Count -eq 0) {
    $attempt++
    [System.IO.File]::WriteAllText($todo, ("[Worker #1]`r`n" + $task + "`r`n"), $enc)
    $sigma = 1000003 * $attempt
    $text = [System.IO.File]::ReadAllText($ini) -replace '(?m)^sigma\s*=.*$', "sigma = $sigma"
    if ($text -notmatch '(?m)^sigma\s*=') { $text = "sigma = $sigma`r`n" + $text }
    [System.IO.File]::WriteAllText($ini, $text, $enc)
    Remove-Item -LiteralPath $logFile -Force -ErrorAction SilentlyContinue
    $out = (& cmd /c $cmd | Out-String)
    $lines = @($out -split "`r?`n")
    $hits = @($lines | Where-Object { $_ -match 'factor\[' })
    $done = @($lines | Where-Object { $_ -match 'queue done, 1 task' })
    Write-Host ("      attempt {0}: sigma={1} hits={2}" -f $attempt, $sigma, $hits.Count)
}
Check "the queue processed the task" ($done.Count -eq 1)
Check "at least one hit line was printed" ($hits.Count -ge 1) `
      ("hits=" + $hits.Count + " after " + $attempt + " attempt(s)")

$log = if (Test-Path $logFile) { [System.IO.File]::ReadAllText($logFile) } else { '' }
Check "the hit lines also reached the per-worker log" `
      ((@($log -split "`r?`n" | Where-Object { $_ -match 'factor\[' })).Count -eq $hits.Count)

Write-Host "[2] every hit line carries the D3 fields"
$re = 'factor\[(\d+)\]=(\d+) curve=(\d+) sigma=(\d+) param=(\d+) method=(\w+) save=(\S*)\s*$'
$parsed = @()
$bad = 0
foreach ($l in $hits) {
    $plain = $l -replace '^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ', ''
    $m = [regex]::Match($plain, $re)
    if (-not $m.Success) { $bad++; continue }
    $parsed += [pscustomobject]@{
        idx    = [int]$m.Groups[1].Value
        factor = $m.Groups[2].Value
        curve  = [int]$m.Groups[3].Value
        sigma  = [System.Numerics.BigInteger]::Parse($m.Groups[4].Value)
        param  = $m.Groups[5].Value
        method = $m.Groups[6].Value
        save   = $m.Groups[7].Value
    }
}
Check "all $($hits.Count) hit line(s) have all six fields" `
      ($bad -eq 0 -and $parsed.Count -eq $hits.Count) ("$bad malformed")
if ($parsed.Count -gt 0) {
    Write-Host ("  first: " + ($hits[0] -replace '^\[[^\]]+\] ', ''))

    Write-Host "[3] the fields are internally consistent"
    Check "curve == the index in factor[i]" (@($parsed | Where-Object { $_.curve -ne $_.idx }).Count -eq 0)
    $bases = @($parsed | ForEach-Object { ($_.sigma - $_.curve).ToString() } | Sort-Object -Unique)
    Check "sigma - curve is one constant (the batch base)" ($bases.Count -eq 1) ($bases -join ',')
    Check "param = 3 (CUDA default parametrization)" (@($parsed | Where-Object { $_.param -ne '3' }).Count -eq 0)
    Check "method = gpu" (@($parsed | Where-Object { $_.method -ne 'gpu' }).Count -eq 0)
    Check ("save = " + $saveName) (@($parsed | Where-Object { $_.save -ne $saveName }).Count -eq 0)

    Write-Host "[4] the reported factors are real"
    $N = [System.Numerics.BigInteger]::Pow(2, $Exp) - 1
    $notDivisor = @($parsed | Where-Object { ($N % [System.Numerics.BigInteger]::Parse($_.factor)) -ne 0 })
    Check ("every factor divides 2^" + $Exp + "-1") ($notDivisor.Count -eq 0) `
          (($notDivisor | ForEach-Object { $_.factor }) -join ',')
    Check "the known M677 factor 1943118631 is among them" `
          (@($parsed | Where-Object { $_.factor -eq '1943118631' }).Count -ge 1)
}

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
