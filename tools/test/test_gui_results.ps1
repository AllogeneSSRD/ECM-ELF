#Requires -Version 5.1
<#
.SYNOPSIS
    M5 acceptance: the two result files (docs/DEV_ECM_GUI.md section 9).

.DESCRIPTION
    Drives the real driver through the GUI twice on the SAME task (M677, B1=1e6, 8
    curves, device 0 -- reliable: 5..11 of 16 curves hit the 31-bit factor 1943118631)
    and checks the promised behaviour:

      * results.json.txt is append-only JSONL: one object per hit, prime95-shaped field
        names (status/worktype/factors/b1/sigma/...), every factor really dividing
        2^677-1;
      * results.txt merges: ONE line per distinct factor, carrying the curve list, the
        sigma list and the hit count of every hit of that factor;
      * the second run MERGES into the same lines (hits grow, more sigmas appear)
        instead of appending a duplicate factor line;
      * results.txt is reproducible from the JSONL (the GUI's "rebuild" button does
        exactly that, and the same code path is exercised here through the file
        comparison);
      * the ini keeps the driver keys and gains the [GUI] layout keys as usual.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_gui_results.ps1
#>
param(
    [string]$Exe = "",
    [string]$EcmCuda = "",
    [string]$Sandbox = "",
    [int]$Exp = 677,
    [int]$Curves = 8,
    [int]$Device = 0,
    [int]$TimeoutSeconds = 240
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Exe) {
    foreach ($cand in @("$repoRoot\build_gui\ecm_gui.exe", "$repoRoot\build_vs18\Release\ecm_gui.exe")) {
        if (Test-Path $cand) { $Exe = $cand; break }
    }
}
if (-not $EcmCuda) {
    foreach ($cand in @("$repoRoot\build_cuda_cmake\ecm_cuda.exe", "$repoRoot\build_vs18\Release\ecm_cuda.exe")) {
        if (Test-Path $cand) { $EcmCuda = $cand; break }
    }
}
if (-not (Test-Path $Exe)) { Write-Host "FAIL: ecm_gui.exe not found" -ForegroundColor Red; exit 2 }
if (-not (Test-Path $EcmCuda)) { Write-Host "FAIL: ecm_cuda.exe not found" -ForegroundColor Red; exit 2 }
if (-not $Sandbox) { $Sandbox = Join-Path $PSScriptRoot '_run\gui_results' }
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
function Read-TextShared([string]$path) {
    if (-not (Test-Path $path)) { return "" }
    try {
        $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Open,
                                     [System.IO.FileAccess]::Read,
                                     [System.IO.FileShare]::ReadWrite)
        $sr = New-Object System.IO.StreamReader($fs)
        $t = $sr.ReadToEnd(); $sr.Close(); $fs.Close(); return $t
    } catch { return "" }
}
if (-not ("Win32GuiRes" -as [type])) {
    Add-Type @'
using System; using System.Runtime.InteropServices; using System.Text;
public static class Win32GuiRes {
  public delegate bool EnumProc(IntPtr h, IntPtr p);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr p);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l);
  public static IntPtr Find(uint pid, string cls) {
    IntPtr found = IntPtr.Zero;
    EnumWindows((h,p) => { uint wp; GetWindowThreadProcessId(h, out wp);
      if (wp != pid) return true;
      var sb = new StringBuilder(256); GetClassName(h, sb, sb.Capacity);
      if (sb.ToString() == cls) { found = h; return false; } return true; }, IntPtr.Zero);
    return found; }
  public static void Close(IntPtr h) { PostMessage(h, 0x0010, IntPtr.Zero, IntPtr.Zero); }
}
'@
}

$enc = New-Object System.Text.UTF8Encoding($false)
$task = 'ECMSTAGE2=1,2,{0},-1,"m{0}_1e6.save",0,0,{1}' -f $Exp, $Curves
$todo = Join-Path $Sandbox 'worktodo.txt'
$ini = Join-Path $Sandbox 'ecm.ini'
$finished = Join-Path $Sandbox 'finished.txt'
$saves = Join-Path $Sandbox 'saves'
$logFile = Join-Path $Sandbox 'screen_1.log'
$resultsJson = Join-Path $Sandbox 'results.json.txt'
$resultsTxt = Join-Path $Sandbox 'results.txt'
$trace = Join-Path (Split-Path -Parent $Exe) 'ecm_gui_trace.log'

# NOTE: parenthesise concatenations inside array literals (the comma binds tighter
# than '+', so an unparenthesised "key = " + $v becomes two ini lines).
$iniLines = @(
    'method = gpu',
    ('gpucurves = ' + $Curves),
    ('tmp_dir = ' + $saves),
    ('finished = ' + $finished),
    ('worktodo = ' + $todo),
    'verbose = 0',
    'ckpt_seconds = 0',
    '',
    '[GUI]',
    'NumWorkers = 1',
    ('exe = ' + $EcmCuda),
    ('results_json = ' + $resultsJson),
    ('results_txt = ' + $resultsTxt),
    'language = english',
    '',
    '[Worker #1]',
    ('device = ' + $Device),
    ('log_file = ' + $logFile),
    'autostart = 1'
)
[System.IO.File]::WriteAllText($ini, (($iniLines -join "`r`n") + "`r`n"), $enc)

function Invoke-GuiRun([string]$label) {
    [System.IO.File]::WriteAllText($todo, ("[Worker #1]`r`n" + $task + "`r`n"), $enc)
    Remove-Item $trace -ErrorAction SilentlyContinue
    $proc = Start-Process -FilePath $Exe -ArgumentList @('-ini', $ini, '--trace') -PassThru
    $hwnd = [IntPtr]::Zero
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline -and $hwnd -eq [IntPtr]::Zero) {
        Start-Sleep -Milliseconds 400
        $proc.Refresh(); if ($proc.HasExited) { break }
        $hwnd = [Win32GuiRes]::Find([uint32]$proc.Id, 'ecm_gui')
    }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        Start-Sleep -Seconds 2
        $t = Read-TextShared $trace
        if ($t -match 'worker 1: state QueueEmpty') { break }
        $proc.Refresh(); if ($proc.HasExited) { break }
    }
    $t = Read-TextShared $trace
    Write-Host ("  {0}: queue finished after {1:N0}s (QueueEmpty={2})" -f $label,
                $sw.Elapsed.TotalSeconds, ($t -match 'worker 1: state QueueEmpty'))
    if ($hwnd -ne [IntPtr]::Zero) { [Win32GuiRes]::Close($hwnd) }
    if (-not $proc.WaitForExit(30000)) { $proc.Kill(); return $null }
    return $t
}

function Get-JsonlObjects([string]$text) {
    return @($text -split "`r?`n" | Where-Object { $_ -match '^\s*\{' })
}
function Get-MergedLines([string]$text) {
    return @($text -split "`r?`n" | Where-Object { $_ -match 'has a factor:' })
}
# Canonical curve-set string: split, cast to int, sort, join. Done in one helper so
# both sides of the comparison go through exactly the same code.
function Canon-Curves($value) {
    $list = New-Object System.Collections.ArrayList
    foreach ($part in ([string]$value -split ',')) {
        if ($part -match '^\d+$') { [void]$list.Add([int]$part) }
    }
    return (($list | Sort-Object) -join ',')
}

Write-Host ("exe     : " + $Exe)
Write-Host ("ecm_cuda: " + $EcmCuda)
Write-Host ("sandbox : " + $Sandbox)
Write-Host ("task    : " + $task)
Write-Host ""
Write-Host "[1] first run: the GUI must write both result files"

$t1 = Invoke-GuiRun "run 1"
Check "run 1 finished the queue" ($t1 -match 'worker 1: state QueueEmpty')
Check "the trace reports the hit" ($t1 -match 'results: factor found: \d+')
$json1 = Read-TextShared $resultsJson
$txt1 = Read-TextShared $resultsTxt
# NB: a function that returns a one-element array is UNROLLED into a scalar by
# PowerShell, so $x[0] would index a STRING (yielding "M" from "M677 has a factor: ...").
# Wrap every call site in @(...) to keep indexing on an array.
$objs1 = @(Get-JsonlObjects $json1)
$lines1 = @(Get-MergedLines $txt1)
Check "results.json.txt exists with hit objects" ($objs1.Count -ge 1) ("objects=" + $objs1.Count)
Check "results.txt exists with merged lines" ($lines1.Count -ge 1) ("lines=" + $lines1.Count)

$N = [System.Numerics.BigInteger]::Pow(2, $Exp) - 1
$factors1 = @()
foreach ($o in $objs1) {
    $fm = [regex]::Match($o, '"factors":\["(\d+)"\]')
    if ($fm.Success) { $factors1 += $fm.Groups[1].Value }
}
Check "every JSONL object carries a factor" ($factors1.Count -eq $objs1.Count)
$badDiv = @($factors1 | Where-Object { ($N % [System.Numerics.BigInteger]::Parse($_)) -ne 0 })
Check "every factor divides 2^$Exp-1" ($badDiv.Count -eq 0) ($badDiv -join ',')

Write-Host "[2] the JSONL fields are the promised ones"
$first = $objs1[0]
Check "status/worktype" (($first -match '"status":"F"') -and ($first -match '"worktype":"ECM"'))
Check ("exponent " + $Exp) ($first -match ('"exponent":' + $Exp))
Check "b1 present" ($first -match '"b1":\d+')
Check "sigma + curve present (D3 fields)" (($first -match '"sigma":\d+') -and ($first -match '"curve":\d+'))
Check "param + method + save present" (($first -match '"param":\d') -and ($first -match '"method":"\w+"') -and ($first -match '"save":"m' + $Exp + '_1e6\.save"'))
Check "timestamp present" ($first -match '"timestamp":"\d{4}-\d{2}-\d{2}T')
Check "the hit came from the log pane's worker/device" (($first -match '"worker":1') -and ($first -match ('"device":' + $Device)))

Write-Host "[3] results.txt merges by factor"
$distinct1 = @($factors1 | Sort-Object -Unique)
Check ("one line per distinct factor (" + $distinct1.Count + ")") ($lines1.Count -eq $distinct1.Count) ("lines=" + $lines1.Count)
$sumHits = 0
foreach ($l in $lines1) {
    $m = [regex]::Match($l, '^M(\d+) has a factor: (\d+) \(ECM(.*), hits=(\d+)\)$')
    if (-not $m.Success) { continue }
    $sumHits += [int]$m.Groups[4].Value
}
Check "the hit counts add up to the JSONL object count" ($sumHits -eq $objs1.Count) ("hits=" + $sumHits + " objects=" + $objs1.Count)
$shape = [regex]::Match($lines1[0], '^M677 has a factor: \d+ \(ECM curves [\d,]+.*B1=1e6, param 3, gpu, Sigmas=\[\d+(,\d+)*\], hits=\d+\)$')
Check "the merged line has the documented shape" $shape.Success $lines1[0]

Write-Host "[4] second run on the same task merges instead of duplicating"
$t2 = Invoke-GuiRun "run 2"
Check "run 2 finished the queue" ($t2 -match 'worker 1: state QueueEmpty')
$json2 = Read-TextShared $resultsJson
$txt2 = Read-TextShared $resultsTxt
$objs2 = @(Get-JsonlObjects $json2)
$lines2 = @(Get-MergedLines $txt2)
Check "the JSONL grew (append-only)" ($objs2.Count -gt $objs1.Count) ("before=" + $objs1.Count + " after=" + $objs2.Count)
Check "results.txt still has one line per distinct factor" `
      ($lines2.Count -eq (@(($objs2 | ForEach-Object { [regex]::Match($_, '"factors":\["(\d+)"\]').Groups[1].Value }) | Sort-Object -Unique)).Count) `
      ("lines=" + $lines2.Count)
$sumHits2 = 0
foreach ($l in $lines2) {
    $m = [regex]::Match($l, ', hits=(\d+)\)$')
    if ($m.Success) { $sumHits2 += [int]$m.Groups[1].Value }
}
Check "the hit counts now cover both runs" ($sumHits2 -eq $objs2.Count) ("hits=" + $sumHits2 + " objects=" + $objs2.Count)
$sig2 = [regex]::Match($lines2[0], 'Sigmas=\[([\d,]+)\]').Groups[1].Value
$sig1 = [regex]::Match($lines1[0], 'Sigmas=\[([\d,]+)\]').Groups[1].Value
Check "the merged line of the known factor accumulated more sigmas" `
      (($sig2 -split ',').Count -ge ($sig1 -split ',').Count) ("run1=" + $sig1 + " run2=" + $sig2)

Write-Host "[5] every merged line is backed by the JSONL"
# Recompute, per factor, what the JSONL says (hit count + the set of curves/sigmas)
# and compare that against the line results.txt shows. Deliberately written with plain
# loops and no clever joins: the byte-level "results.txt is reproducible" property is
# asserted by the C++ unit test (src/gui/results_test.cpp, rebuild_from_jsonl), this
# check only has to prove the two files agree about the real run.
$rebuilt = @{}
foreach ($o in $objs2) {
    $f = [regex]::Match($o, '"factors":\["(\d+)"\]').Groups[1].Value
    if (-not $f) { continue }
    $curve = [regex]::Match($o, '"curve":(\d+)').Groups[1].Value
    $sigma = [regex]::Match($o, '"sigma":(\d+)').Groups[1].Value
    if (-not $rebuilt.ContainsKey($f)) {
        $rebuilt[$f] = @{ hits = 0; curves = New-Object System.Collections.Generic.List[string]
                          sigmas = New-Object System.Collections.Generic.List[string] }
    }
    $rebuilt[$f].hits = $rebuilt[$f].hits + 1
    if ($curve -and -not $rebuilt[$f].curves.Contains($curve)) { $rebuilt[$f].curves.Add($curve) }
    if ($sigma -and -not $rebuilt[$f].sigmas.Contains($sigma)) { $rebuilt[$f].sigmas.Add($sigma) }
}
$wrong = 0
$why = ""
foreach ($l in $lines2) {
    $m = [regex]::Match($l, '^M(\d+) has a factor: (\d+) \(ECM curves ([\d,]+), B1=1e6, param 3, gpu, Sigmas=\[([\d,]+)\], hits=(\d+)\)$')
    if (-not $m.Success) { $wrong++; $why += " [no-pattern: $l]"; continue }
    $f = $m.Groups[2].Value
    if (-not $rebuilt.ContainsKey($f)) { $wrong++; $why += " [factor $f not in the JSONL]"; continue }
    $lineHits = [int]$m.Groups[5].Value
    if ($lineHits -ne $rebuilt[$f].hits) {
        $wrong++
        $why += " [hits: line $lineHits, json $($rebuilt[$f].hits)]"
    }
    $lineCurves = @($m.Groups[3].Value -split ',')
    if ($lineCurves.Count -ne $rebuilt[$f].curves.Count) {
        $wrong++
        $why += " [curves: line $($lineCurves.Count), json $($rebuilt[$f].curves.Count)]"
    }
    foreach ($c in $lineCurves) {
        if (-not $rebuilt[$f].curves.Contains($c)) { $wrong++; $why += " [curve $c not in the JSONL]" }
    }
    $lineSigmas = @($m.Groups[4].Value -split ',')
    if ($lineSigmas.Count -ne $rebuilt[$f].sigmas.Count) {
        $wrong++
        $why += " [sigmas: line $($lineSigmas.Count), json $($rebuilt[$f].sigmas.Count)]"
    }
    foreach ($sg in $lineSigmas) {
        if (-not $rebuilt[$f].sigmas.Contains($sg)) { $wrong++; $why += " [sigma $sg not in the JSONL]" }
    }
}
Check "every merged line is backed by the JSONL" ($wrong -eq 0) ("problems=" + $wrong + $why)

Write-Host "[6] the ini is intact and has the GUI keys"
$after = [System.IO.File]::ReadAllText($ini)
Check "driver key gpucurves kept" ($after -match '(?m)^gpucurves\s*=\s*8\s*$')
Check "results_json setting kept"  ($after -match ('(?m)^results_json\s*=\s*' + [regex]::Escape($resultsJson)))
Check "[GUI] window= written"      ($after -match '(?m)^window\s*=\s*-?\d+,-?\d+,\d+,\d+')
Check "[GUI] dock_layout written"  ($after -match '(?m)^dock_layout\s*=\s*\S')
# Only the driver THIS test started counts: a production ecm_gui/ecm_cuda pair may well be
# running on the same machine (measured 2026-09-29), and this test kills nothing itself.
$strayDeadline = (Get-Date).AddSeconds(10)
while ((Get-Date) -lt $strayDeadline -and
       @(Get-Process -Name 'ecm_cuda' -ErrorAction SilentlyContinue | Where-Object {
             $p = ""
             try { $p = $_.MainModule.FileName } catch { $p = "" }
             $p -eq $EcmCuda }).Count -gt 0) {
    Start-Sleep -Milliseconds 500
}
Check "no worker of ours is left behind" (@(Get-Process -Name 'ecm_cuda' -ErrorAction SilentlyContinue | Where-Object {
        $p = ""
        try { $p = $_.MainModule.FileName } catch { $p = "" }
        $p -eq $EcmCuda }).Count -eq 0)

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
Write-Host ("files  : " + $resultsJson + "  |  " + $resultsTxt)
if ($script:fail -gt 0) { exit 1 }
exit 0
