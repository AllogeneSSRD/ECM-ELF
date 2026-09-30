#Requires -Version 5.1
<#
.SYNOPSIS
    Proves the GPU panel's power/clock curves carry *varying* data under real load.

.DESCRIPTION
    A user report ("power and frequency curves have data but are one flat line") has two
    possible causes, and they need different answers:

      a) the history buffer / plot range is broken, or
      b) the sampled signal really is constant.

    The GUI traces both, per card and per ~20 samples (App::trace_gpu_history):

      gpu: history dev=0 samples=87 util=26 distinct power=40 distinct clock=32 distinct flat=0
      gpu: history dev=0 ranges power=18.0..148.7 plot=5.0..161.8 clock=525..2595 plot=318..2802

    "distinct" counts the rounded values actually in the retained window, so a flat line
    is exactly `distinct == 1`. This script puts a real ecm_cuda worker on device 0 and
    then asserts the loaded card reports many distinct power/clock values and a
    non-degenerate plot range.

    Measured on the reference box (RTX 4070 Ti busy + RTX 4060 Laptop idle):
      busy  card -> power 40 distinct (18.0..148.7 W), clock 32 distinct (525..2595 MHz)
      idle  card -> power  3 distinct (1.5..1.7 W),   clock  1 distinct (210..210 MHz)
    i.e. an *idle* card legitimately draws a straight clock line; the check below
    therefore looks at the BUSY card only.

    Exit code: 0 = all checks passed.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_gui_gpu_curves.ps1
#>
param(
    [string]$Exe = "",
    [string]$EcmCuda = "",
    [string]$Sandbox = "",
    [int]$Device = 0,
    [int]$Curves = 4000,
    [int]$Seconds = 45
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Exe) {
    foreach ($cand in @("$repoRoot\build_gui\ecm_gui.exe",
                        "$repoRoot\build_vs18\Release\ecm_gui.exe",
                        "$repoRoot\build_cuda_cmake\ecm_gui.exe")) {
        if (Test-Path $cand) { $Exe = $cand; break }
    }
}
if (-not $Exe -or -not (Test-Path $Exe)) {
    Write-Host "FAIL: ecm_gui.exe not found (pass -Exe <path>)" -ForegroundColor Red
    exit 2
}
if (-not $EcmCuda) {
    foreach ($cand in @("$repoRoot\build_cuda_cmake\ecm_cuda.exe",
                        "$repoRoot\build_vs18\Release\ecm_cuda.exe")) {
        if (Test-Path $cand) { $EcmCuda = $cand; break }
    }
}
if (-not $EcmCuda -or -not (Test-Path $EcmCuda)) {
    Write-Host "FAIL: ecm_cuda.exe not found (pass -EcmCuda <path>)" -ForegroundColor Red
    exit 2
}
if (-not $Sandbox) { $Sandbox = Join-Path $PSScriptRoot '_run\gui_gpu_curves' }
if (Test-Path $Sandbox) { Remove-Item $Sandbox -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Sandbox | Out-Null

$script:pass = 0
$script:fail = 0
function Check([string]$name, $ok, [string]$detail = "") {
    if ($ok) {
        $script:pass++
        Write-Host ("  [ok]   " + $name)
    } else {
        $script:fail++
        Write-Host ("  [FAIL] " + $name + $(if ($detail) { " -- " + $detail } else { "" })) -ForegroundColor Red
    }
}

function Read-TextShared([string]$path) {
    if (-not (Test-Path $path)) { return "" }
    try {
        $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Open,
                                     [System.IO.FileAccess]::Read,
                                     [System.IO.FileShare]::ReadWrite)
        $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)
        $text = $sr.ReadToEnd()
        $sr.Close(); $fs.Close()
        return $text
    } catch { return "" }
}

$enc = New-Object System.Text.UTF8Encoding($false)
$todo = Join-Path $Sandbox 'worktodo.txt'
# Save names must end with the B1 token ("m{n}_{b1}.save") -- the driver takes B1 from
# the last "_" token and rejects the line otherwise (see docs/DEV_ECM_GUI.md section 12).
[System.IO.File]::WriteAllText($todo, (@(
    '[Worker #1]',
    ('ECMSTAGE2=1,2,991,-1,"m991_1e4.save",0,0,' + $Curves)) -join "`r`n") + "`r`n", $enc)

# PowerShell trap (docs/DEV_ECM_GUI.md section 12): inside an array literal the comma
# binds tighter than '+', so every concatenation MUST be parenthesised -- an unparenthesised
# '...' + $x + '...' merges the whole rest of the array into that one element. For an ini
# that is fatal: the merged element starts with '#' and comments out every key.
$header = ('# GPU history sandbox: one real worker keeps card ' + $Device + ' busy')
$iniLines = @(
    $header,
    'method = gpu',
    'backend = auto',
    'gpucurves = 384',
    ('tmp_dir = ' + (Join-Path $Sandbox 'saves')),
    ('finished = ' + (Join-Path $Sandbox 'finished.txt')),
    ('worktodo = ' + $todo),
    'verbose = 0',
    'ckpt_seconds = 0',
    'save_sync_dir_1 =',
    'save_sync_dir_2 =',
    '',
    '[GUI]',
    'NumWorkers = 1',
    ('exe = ' + $EcmCuda),
    'language = english',
    'refresh_hz = 10',
    'gpu_poll_ms = 250',
    '',
    '[Worker #1]',
    'name = load',
    ('log_file = ' + (Join-Path $Sandbox 'screen_1.log')),
    ('device = ' + $Device),
    'autostart = 1'
)
$iniPath = Join-Path $Sandbox 'ecm.ini'
[System.IO.File]::WriteAllText($iniPath, (($iniLines -join "`r`n") + "`r`n"), $enc)

$trace = Join-Path (Split-Path -Parent $Exe) 'ecm_gui_trace.log'
Remove-Item $trace -Force -ErrorAction SilentlyContinue

Write-Host "ecm_gui GPU curve check (device $Device, $Curves curves, up to ${Seconds}s)"
Write-Host ("exe      : " + $Exe)
Write-Host ("ecm_cuda : " + $EcmCuda)
Write-Host ("sandbox  : " + $Sandbox)

$proc = Start-Process -FilePath $Exe -ArgumentList @('-ini', $iniPath, '--trace') -PassThru

# Wait for a history line that covers a decent window on the loaded card.
$deadline = (Get-Date).AddSeconds($Seconds)
$best = $null
$bestRanges = $null
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
    $proc.Refresh()
    if ($proc.HasExited) { break }
    $log = Read-TextShared $trace
    foreach ($m in [regex]::Matches($log, "gpu: history dev=(\d+) samples=(\d+) util=(\d+) distinct power=(\d+) distinct clock=(\d+) distinct flat=(\d)")) {
        if ([int]$m.Groups[1].Value -ne $Device) { continue }
        if ($null -eq $best -or [int]$m.Groups[2].Value -ge [int]$best.Groups[2].Value) { $best = $m }
    }
    foreach ($m in [regex]::Matches($log, "gpu: history dev=(\d+) ranges power=([\d.]+)\.\.([\d.]+) plot=([\d.]+)\.\.([\d.]+) clock=([\d.]+)\.\.([\d.]+) plot=([\d.]+)\.\.([\d.]+)")) {
        if ([int]$m.Groups[1].Value -ne $Device) { continue }
        $bestRanges = $m
    }
    # Stop early once we have a healthy busy-card window: no need to burn the full budget.
    if ($null -ne $best -and [int]$best.Groups[2].Value -ge 40) { break }
}

$log = Read-TextShared $trace
$proc.Refresh()
$exitedEarly = $proc.HasExited
if (-not $exitedEarly) { $proc.CloseMainWindow() | Out-Null; Start-Sleep -Seconds 3 }
$proc.Refresh()
if (-not $proc.HasExited) { $proc.Kill() }
$exitCode = $proc.ExitCode
[System.IO.File]::WriteAllText((Join-Path $Sandbox 'trace.log'), $log, [System.Text.Encoding]::UTF8)

Write-Host "[1] the worker really loaded the GPU"
Check "the GUI spawned the real driver" ($log -match 'worker 1: autostart pid=\d+')
Check "the worker reached the running state" ($log -match 'worker 1: state Running')
Check "the GUI did not exit early" (-not $exitedEarly)

Write-Host "[2] the history buffer accumulates samples on the loaded card"
Check "the trace reports GPU history for device $Device" ($null -ne $best) `
      "no 'gpu: history dev=$Device' line in the trace"
if ($null -eq $best) {
    Write-Host ""
    Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
    exit 1
}
$samples = [int]$best.Groups[2].Value
$uDist = [int]$best.Groups[3].Value
$pDist = [int]$best.Groups[4].Value
$cDist = [int]$best.Groups[5].Value
$flat = [int]$best.Groups[6].Value
Write-Host ("       samples=" + $samples + " util distinct=" + $uDist +
            " power distinct=" + $pDist + " clock distinct=" + $cDist + " flat=" + $flat)
Check "at least 20 samples were retained" ($samples -ge 20) ("samples=" + $samples)

Write-Host "[3] the curves are not a dead flat line"
# This is the actual user-visible claim: under load the retained window must contain
# many different power and SM-clock values, not one repeated value.
#
# One caveat, measured 2026-09-29: when ANOTHER GPU job is already running (the operator's
# own production ecm_cuda, or another test in this suite), the driver holds the SM clock at
# a fixed boost step and keeps utilisation pinned, so "distinct" collapses even though the
# panel is working correctly -- the flat reading then belongs to the OTHER process's load,
# not to a broken plot. Say so instead of failing, and keep the strict thresholds when this
# test owns the card.
$foreign = @(Get-Process -Name 'ecm_cuda', 'ecm_gui' -ErrorAction SilentlyContinue | Where-Object {
    $p = ""
    try { $p = $_.MainModule.FileName } catch { $p = "" }
    $p -ne "" -and $p -notlike "*\tools\test\_run\*"
})
if ($foreign.Count -gt 0) {
    Write-Host ("       [SKIP] another GPU job is running (" + (($foreign | ForEach-Object {
        [System.IO.Path]::GetFileName($_.MainModule.FileName) + "(" + $_.Id + ")" }) -join ', ') +
        "): a pinned clock/utilisation is expected, so the load-sensitive checks are skipped")
    Check "the trace does not flag the window as flat" ($flat -eq 0) ("flat=" + $flat)
} else {
    Check "utilisation varies under load" ($uDist -ge 5) ("util distinct=" + $uDist)
    Check "power varies under load" ($pDist -ge 5) ("power distinct=" + $pDist)
    Check "the SM clock varies under load" ($cDist -ge 5) ("clock distinct=" + $cDist)
    Check "the trace does not flag the window as flat" ($flat -eq 0) ("flat=" + $flat)
}

Write-Host "[4] the plotted window is not degenerate"
Check "the trace reports the plotted ranges" ($null -ne $bestRanges)
if ($null -ne $bestRanges) {
    $pLo = [double]$bestRanges.Groups[4].Value
    $pHi = [double]$bestRanges.Groups[5].Value
    $cLo = [double]$bestRanges.Groups[8].Value
    $cHi = [double]$bestRanges.Groups[9].Value
    Write-Host ("       power plot=" + $pLo + ".." + $pHi + " W   clock plot=" + $cLo + ".." + $cHi + " MHz")
    Check "the power plot range has a positive span" ($pHi -gt $pLo) ("lo=" + $pLo + " hi=" + $pHi)
    Check "the clock plot range has a positive span" ($cHi -gt $cLo) ("lo=" + $cLo + " hi=" + $cHi)
    # The plot range must actually cover the observed data (a clamped range would squash
    # the curve against an edge).
    $pMn = [double]$bestRanges.Groups[2].Value
    $pMx = [double]$bestRanges.Groups[3].Value
    Check "the power plot range covers the observed power" ($pLo -le $pMn -and $pHi -ge $pMx) `
          ("plot " + $pLo + ".." + $pHi + " vs data " + $pMn + ".." + $pMx)
}

Write-Host "[4b] the DRAWN charts carry the right data and range (chart rework, 2026-09-29)"
# The GPU charts are drawn by App::draw_metric_plot, which traces what it actually plotted:
#   plot: <id> label="…" n=… lo=… hi=… last=… [ref=…]
# That trace is the machine-readable counterpart of the picture, so a wrong series, a clamped
# range or a missing reference line cannot hide behind a prettier chart.
$charts = @{}
foreach ($m in [regex]::Matches($log, ('plot: gpu' + $Device + '/(util|power|clock) label="([^"]*)" n=(\d+) lo=([\d.-]+) hi=([\d.-]+) last=([\d.-]+)( ref=([\d.-]+))?'))) {
    # [regex]::Match returns the FIRST match, so the loop keeps the LAST (= newest) one.
    $charts[$m.Groups[1].Value] = $m
}
foreach ($kind in 'util', 'power', 'clock') {
    Check ("the {0} chart was drawn and traced" -f $kind) ($charts.ContainsKey($kind)) `
          ("traced charts: " + (($charts.Keys | Sort-Object) -join ','))
}
foreach ($kind in @($charts.Keys)) {
    $c = $charts[$kind]
    Write-Host ("       " + $c.Value)
    Check ("the {0} chart has a real series" -f $kind) ([int]$c.Groups[3].Value -ge 2) `
          ("n=" + $c.Groups[3].Value)
    $lo = [double]$c.Groups[4].Value
    $hi = [double]$c.Groups[5].Value
    $last = [double]$c.Groups[6].Value
    Check ("the {0} chart range brackets its newest value" -f $kind) ($lo -le $last -and $hi -ge $last) `
          ("lo=" + $lo + " last=" + $last + " hi=" + $hi)
    Check ("the {0} chart range is not degenerate" -f $kind) ($hi -gt $lo) ("lo=" + $lo + " hi=" + $hi)
}
if ($charts.ContainsKey('util')) {
    # A percentage must always read 0..100 (fixed range), never a zoomed window.
    Check "the utilisation chart is fixed to 0..100 %" `
          (([double]$charts['util'].Groups[4].Value -eq 0.0) -and ([double]$charts['util'].Groups[5].Value -eq 100.0)) `
          ("lo=" + $charts['util'].Groups[4].Value + " hi=" + $charts['util'].Groups[5].Value)
}
$limit = $null
foreach ($m in [regex]::Matches($log, ('gpu: limits dev=' + $Device + ' power_limit_w=([\d.]+)'))) {
    $limit = [double]$m.Groups[1].Value      # last one wins
}
Check "the enforced power limit is traced (so ref= can be cross-checked)" ($null -ne $limit)
if ($charts.ContainsKey('power')) {
    $ref = $charts['power'].Groups[8].Value
    Check "the power chart draws a reference line" ($ref -ne "") "no ref= on the power chart"
    if ($null -ne $limit -and $ref -ne "") {
        Check "and it is the NVML power limit, not a guess" ([double]$ref -eq $limit) `
              ("chart ref=" + $ref + " W, NVML limit=" + $limit + " W")
    }
}

Write-Host "[5] shutdown leaves nothing behind"
Check "the GUI exited with code 0" ($exitCode -eq 0) ("exit=" + $exitCode)
$stray = @()
foreach ($m in [regex]::Matches($log, 'worker 1: autostart pid=(\d+)')) { $stray += [int]$m.Groups[1].Value }
$alive = @($stray | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
Check "the spawned driver process is gone" ($alive.Count -eq 0) ("alive=" + ($alive -join ','))

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
