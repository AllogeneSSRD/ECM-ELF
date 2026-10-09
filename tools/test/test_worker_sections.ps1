#Requires -Version 5.1
<#
.SYNOPSIS
    Acceptance test for D1 (ini [Worker #N] sections) and D2 (worktodo [Worker #N]
    sections) -- see docs/usage/GUI.md and docs/usage/STAGE1.md.

.DESCRIPTION
    Builds a throw-away sandbox with

        ecm.ini      : global keys + [Worker #1] + [Worker #2]
        worktodo.txt : [Worker #1] + [Worker #2], each holding one UNPARSEABLE task
                       line (so the driver never touches the GPU: a bad line is
                       marked "# ERROR <line>" and the queue keeps going)

    and then runs the queue manager three times:

        1) --worker 1 : must use device 0, log to screen.log, and mark ONLY the
                        line of [Worker #1]
        2) --worker 2 : must use device 1 (section override), log to screen_2.log
                        (per-worker default), and mark ONLY the line of [Worker #2]
        3) no sections in worktodo, no --worker : legacy behaviour (worker 1, whole
                        file is the queue) -- the regression guarantee

    Each check is printed; the script exits non-zero if any check failed.
    ASCII only on purpose (Windows PowerShell 5.1 needs a BOM for non-ASCII .ps1).

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_worker_sections.ps1
#>
param(
    [string]$Exe = "",
    [string]$Sandbox = ""
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Exe) {
    foreach ($cand in @("$repoRoot\build_cuda_cmake\ecm_cuda.exe",
                        "$repoRoot\build_vs18\Release\ecm_cuda.exe",
                        "$repoRoot\build_cuda_cmake\ecm.exe")) {
        if (Test-Path $cand) { $Exe = $cand; break }
    }
}
if (-not $Exe -or -not (Test-Path $Exe)) {
    Write-Host "FAIL: no ecm/ecm_cuda executable found (pass -Exe <path>)" -ForegroundColor Red
    exit 2
}
if (-not $Sandbox) { $Sandbox = Join-Path $PSScriptRoot '_run\worker_sections' }
if (Test-Path $Sandbox) { Remove-Item $Sandbox -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Sandbox | Out-Null

$script:pass = 0
$script:fail = 0
function Check([string]$name, [bool]$ok, [string]$detail = "") {
    if ($ok) {
        $script:pass++
        Write-Host ("  [ok]   " + $name)
    } else {
        $script:fail++
        Write-Host ("  [FAIL] " + $name + $(if ($detail) { " -- " + $detail } else { "" })) -ForegroundColor Red
    }
}

function Write-Utf8NoBom([string]$path, [string[]]$lines) {
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($path, (($lines -join "`r`n") + "`r`n"), $enc)
}

function Run-Queue([string]$ini, [int]$worker, [bool]$withWorkerFlag) {
    if ($withWorkerFlag) {
        $cmd = '"{0}" -ini "{1}" --worker {2} < NUL 2>&1' -f $Exe, $ini, $worker
    } else {
        $cmd = '"{0}" -ini "{1}" < NUL 2>&1' -f $Exe, $ini
    }
    return (& cmd /c $cmd | Out-String)
}

# Task lines that FAIL TO PARSE ("not enough fields"): the driver marks them
# "# ERROR <line>" and moves on, so the test never touches the GPU.
# Do NOT use something like 'ECMSTAGE2=BOGUS,1,2,5351,-1,"m5351_110e6.save",...':
# the parser treats a non-integer first field as the optional AID, so that line is
# a REAL M5351 task and would start hours of GPU work (this burned us once).
$badW1 = 'ECMSTAGE2=garbage-W1'
$badW2 = 'ECMSTAGE2=garbage-W2'
$iniPath = Join-Path $Sandbox 'ecm.ini'
$todoPath = Join-Path $Sandbox 'worktodo.txt'
$finPath = Join-Path $Sandbox 'finished.txt'
$tmpDir = Join-Path $Sandbox 'saves'
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

# NOTE: inside a PowerShell array literal the comma binds tighter than '+', so
# every concatenation below must be parenthesised (otherwise it becomes two
# elements -- and two ini lines).
Write-Utf8NoBom $iniPath @(
    '# global defaults (all workers)',
    ('tmp_dir = ' + $tmpDir),
    ('worktodo = ' + $todoPath),
    ('finished = ' + $finPath),
    'device = 0',
    'method = gpu',
    '',
    '[GUI]',
    'refresh_hz = 10',
    '',
    '[Worker #1]',
    'gpucurves = 8',
    '',
    '[Worker #2]',
    'device = 1',
    'gpucurves = 16'
)
$iniBefore = Get-Content -LiteralPath $iniPath -Raw

Write-Utf8NoBom $todoPath @(
    '# two sections, each with one broken line',
    '[Worker #1]',
    $badW1,
    '',
    '[Worker #2]',
    $badW2
)

Write-Host ("exe     : " + $Exe)
Write-Host ("sandbox : " + $Sandbox)
Write-Host ""

# ---------------------------------------------------------------- worker 1 ----
Write-Host "[1] --worker 1 (global device=0, section adds gpucurves=8)"
$out1 = Run-Queue $iniPath 1 $true
$todo1 = Get-Content -LiteralPath $todoPath -Raw
Check "worker banner says 1"        ($out1 -match 'worker : 1 ') $out1
Check "parse error reported"        ($out1 -match 'not enough fields') $out1
Check "no task was run (0 tasks)"   ($out1 -match '0 task\(s\) processed') $out1
Check "global device 0 in effect"   ($out1 -match 'device 0')   $out1
Check "log_file stays screen.log"   ($out1 -match 'log_file : screen\.log') $out1
Check "worker 1 line marked"        ($todo1 -match ('# ERROR ' + [regex]::Escape($badW1)))
Check "worker 2 line untouched"     ($todo1 -match [regex]::Escape($badW2))
Check "no ERROR prefix on W2 line"  (-not ($todo1 -match ('# ERROR ' + [regex]::Escape($badW2))))
Check "both section headers kept"   (($todo1 -match '\[Worker #1\]') -and ($todo1 -match '\[Worker #2\]'))
Check "comment kept"                ($todo1 -match '# two sections, each with one broken line')
Check "ini not modified"            ((Get-Content -LiteralPath $iniPath -Raw) -eq $iniBefore)

# ---------------------------------------------------------------- worker 2 ----
Write-Host "[2] --worker 2 (section overrides device=1, per-worker log default)"
$out2 = Run-Queue $iniPath 2 $true
$todo2 = Get-Content -LiteralPath $todoPath -Raw
Check "worker banner says 2"        ($out2 -match 'worker : 2 ') $out2
Check "section device 1 in effect"  ($out2 -match 'device 1')   $out2
Check "per-worker log screen_2.log" ($out2 -match 'log_file : screen_2\.log') $out2
Check "worker 2 line marked"        ($todo2 -match ('# ERROR ' + [regex]::Escape($badW2)))
Check "worker 1 ERROR line kept"    ($todo2 -match ('# ERROR ' + [regex]::Escape($badW1)))
Check "no W1 task left to run"      (($todo2 -split "`r?`n" | Where-Object { $_ -match 'garbage-W1' }).Count -eq 1)

# ------------------------------------------------- legacy: no sections, no flag ----
Write-Host "[3] no sections, no --worker (legacy single-worker behaviour)"
Write-Utf8NoBom $todoPath @('# plain queue', $badW1)
$out3 = Run-Queue $iniPath 1 $false
$todo3 = Get-Content -LiteralPath $todoPath -Raw
Check "defaults to worker 1"        ($out3 -match 'worker : 1 ') $out3
Check "global device 0 in effect"   ($out3 -match 'device 0')   $out3
Check "plain line marked"           ($todo3 -match ('# ERROR ' + [regex]::Escape($badW1)))
Check "comment kept"                ($todo3 -match '# plain queue')
Check "no [Worker] section needed"  (-not ($todo3 -match '\[Worker #'))

# ------------------------------------------------------- missing section note ----
Write-Host "[4] --worker 3 with no [Worker #3] section (must warn, must not fail)"
Write-Utf8NoBom $todoPath @('[Worker #1]', $badW1)
$out4 = Run-Queue $iniPath 3 $true
Check "note about the missing section" ($out4 -match 'no \[Worker #3\] section')
Check "still uses global device 0"     ($out4 -match 'device 0') $out4

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
