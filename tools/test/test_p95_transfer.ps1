#Requires -Version 5.1
<#
.SYNOPSIS
    Acceptance test for the Prime95 handoff: every finished task is appended VERBATIM to
    the worktodo.add next to Prime95's worktodo.txt (docs/DEV_ECM_GUI.md 18,
    docs/DEV_ECM_WORKTODO.md 8).

.DESCRIPTION
    Each scenario runs ONE real queue-mode task (M521, B1=1e3, 1 curve, param0) in a
    sandbox that mimics a Prime95 directory (worktodo.txt with [Worker #N] sections,
    prime.txt with NumWorkers) and inspects the worktodo.add the driver produced:

      [1] routing by number : p95_add_workers = 2 + a [Worker #2] section -> the line lands
                              under that header, "p95_add: ok worker=2", nothing else touched
      [2] missing section   : p95_add_workers = 3 without [Worker #3] -> header-less append
                              plus a "warn" notice (the GUI shows it yellow)
      [3] auto              : NumWorkers=2 from prime.txt, worker 1 busy -> worker 2 wins
      [4] range             : p95_add_workers = 1-2, worker 1 empty -> worker 1 wins
      [5] invalid specs     : 2000 and "abc" -> warning + header-less append (never a throw)
      [6] verbatim          : AID and known-factors field survive byte for byte, and the
                              line is delivered even when the run finds a factor
      [7] lock held         : a fresh worktodo.add.lock fails the delivery; the line is
                              PARKED and the next task re-delivers it (then pending is gone)
      [8] stale lock        : a lock older than 60 s is preempted instead of blocking

    ASCII only on purpose (Windows PowerShell 5.1 needs a BOM for non-ASCII .ps1).

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_p95_transfer.ps1
#>
param(
    [string]$Exe = "",
    [string]$Sandbox = "",
    [int]$Device = 0
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Exe) {
    foreach ($cand in @("$repoRoot\build_cuda_cmake\ecm_cuda.exe",
                        "$repoRoot\build_vs18\Release\ecm_cuda.exe")) {
        if (Test-Path $cand) { $Exe = $cand; break }
    }
}
if (-not $Exe -or -not (Test-Path $Exe)) {
    Write-Host "FAIL: no ecm_cuda executable found (pass -Exe <path>)" -ForegroundColor Red
    exit 2
}
if (-not $Sandbox) { $Sandbox = Join-Path $PSScriptRoot '_run\p95_transfer' }
if (Test-Path $Sandbox) { Remove-Item $Sandbox -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Sandbox | Out-Null

# The driver parks undelivered lines next to ITS OWN executable.
$pending = Join-Path (Split-Path -Parent $Exe) 'p95_add_pending.txt'
Remove-Item -LiteralPath $pending -Force -ErrorAction SilentlyContinue

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

$enc = New-Object System.Text.UTF8Encoding($false)
function Write-Text([string]$path, [string]$text) {
    [System.IO.File]::WriteAllText($path, $text, $enc)
}

# One scenario: build the sandbox, create the optional lock, run the queue once, collect.
function Invoke-Task {
    param(
        [string]$name,
        [string]$task,
        [string]$workerSpec,
        [string]$p95Todo = "",            # Prime95's worktodo.txt as it should look BEFORE
        [string]$primeTxt = "NumWorkers=2`r`n",
        [int]$lockAgeMinutes = -1,        # -1 = no lock; 0 = fresh lock; >0 = stale lock
        [switch]$KeepAdd                  # keep an existing worktodo.add (for scenario 7b)
    )
    $dir = Join-Path $Sandbox $name
    if (-not $KeepAdd) {
        if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $dir 'p95') | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $dir 'saves') | Out-Null

    $p95TodoPath = Join-Path $dir 'p95\worktodo.txt'
    $todo = Join-Path $dir 'worktodo.txt'
    $ini = Join-Path $dir 'ecm.ini'
    $log = Join-Path $dir 'screen.log'
    $addPath = Join-Path $dir 'p95\worktodo.add'

    Write-Text (Join-Path $dir 'p95\prime.txt') $primeTxt
    Write-Text $p95TodoPath $p95Todo
    Write-Text $todo ("[Worker #1]`r`n" + $task + "`r`n")

    # NOTE: parenthesise every concatenation inside an array literal -- the comma binds
    # tighter than '+', so "key = " + $v would silently become two ini lines.
    $iniLines = @(
        'method = gpu',
        'gpu_param = 0',
        'gpucurves = 1',
        ('device = ' + $Device),
        ('tmp_dir = ' + (Join-Path $dir 'saves')),
        ('finished = ' + (Join-Path $dir 'finished.txt')),
        ('worktodo = ' + $todo),
        ('log_file = ' + $log),
        'verbose = 0',
        'ckpt_seconds = 0',
        ('p95_worktodo_path = ' + $p95TodoPath),
        ('p95_add_workers = ' + $workerSpec)
    )
    Write-Text $ini (($iniLines -join "`r`n") + "`r`n")

    $lockPath = $addPath + '.lock'
    if ($lockAgeMinutes -ge 0) {
        Set-Content -LiteralPath $lockPath -Value 'held by the test' -Encoding ASCII
        if ($lockAgeMinutes -gt 0) {
            (Get-Item -LiteralPath $lockPath).LastWriteTime = (Get-Date).AddMinutes(-$lockAgeMinutes)
        }
    }

    $p95Before = if (Test-Path $p95TodoPath) { Get-Content -LiteralPath $p95TodoPath -Raw } else { '' }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $out = (& cmd /c ('"{0}" -ini "{1}" --worker 1 < NUL 2>&1' -f $Exe, $ini) | Out-String)
    $sw.Stop()

    return @{
        dir         = $dir
        out         = $out
        addPath     = $addPath
        add         = @(if (Test-Path $addPath) { Get-Content -LiteralPath $addPath } else { @() })
        p95TodoPath = $p95TodoPath
        p95Before   = $p95Before
        p95After    = if (Test-Path $p95TodoPath) { Get-Content -LiteralPath $p95TodoPath -Raw } else { '' }
        lockPath    = $lockPath
        seconds     = $sw.Elapsed.TotalSeconds
    }
}

function Get-Notice([string]$out, [string]$kind) {
    return @($out -split "`r?`n" | Where-Object { $_ -match ("p95_add: " + $kind + " ") })[-1]
}
function Get-Index([string[]]$lines, [string]$needle) {
    for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -eq $needle) { return $i } }
    return -1
}

$task = 'ECMSTAGE2=1,2,521,-1,"m521_1e3.save",0,0,1'
$todoBoth = "[Worker #1]`r`nfoo=bar`r`n`r`n[Worker #2]`r`nbaz=qux`r`n"
Write-Host ("driver  : " + $Exe)
Write-Host ("task    : " + $task + " (2^521-1, B1=1e3, 1 curve, device $Device)")
Write-Host ("sandbox : " + $Sandbox)
Write-Host ""

# ------------------------------------------------------------------ [1] routing ---------
Write-Host "[1] p95_add_workers = 2, [Worker #2] exists"
$r1 = Invoke-Task -name 'route2' -task $task -workerSpec '2' -p95Todo $todoBoth
Write-Host ("      elapsed={0:N1}s  add={1}" -f $r1.seconds, ($r1.add -join ' | '))
Check "worktodo.add was created"            (Test-Path $r1.addPath)
Check "the line is under [Worker #2]"       ((Get-Index $r1.add '[Worker #2]') -ge 0 -and $r1.add -contains $task)
Check "the header precedes the line"        ((Get-Index $r1.add '[Worker #2]') -lt (Get-Index $r1.add $task))
Check "no warning when the section exists"  ((Get-Notice $r1.out 'ok') -ne $null -and -not ($r1.out -match 'p95_add: warn'))
Check "the notice says ok worker=2"         ((Get-Notice $r1.out 'ok') -match 'p95_add: ok worker=2 ')
Check "Prime95's worktodo.txt is untouched" ($r1.p95After -eq $r1.p95Before)
Check "the task left our queue"             (-not ((Get-Content (Join-Path $r1.dir 'worktodo.txt') -Raw) -match 'ECMSTAGE2'))
Check "no temp file is left behind"         (-not (Test-Path ($r1.addPath + '.ecm.tmp')))

# ------------------------------------------------------------------ [2] missing ---------
Write-Host "[2] p95_add_workers = 3 without a [Worker #3] section"
$r2 = Invoke-Task -name 'route3' -task $task -workerSpec '3' -p95Todo $todoBoth
$warn2 = Get-Notice $r2.out 'warn'
Check "the line was still delivered"        ($r2.add -contains $task)
Check "no [Worker #3] header was invented"  (@($r2.add | Where-Object { $_ -match '\[Worker #3\]' }).Count -eq 0)
Check "the notice is a warning"             ($null -ne $warn2)
Check "the warning says worker=0"           ($warn2 -match 'worker=0')
Check "the warning names the section"       ($warn2 -match 'section')

# ------------------------------------------------------------------ [3] auto -------------
Write-Host "[3] p95_add_workers = auto (NumWorkers=2, worker 1 has 3 lines)"
$busy1 = "[Worker #1]`r`na=1`r`nb=2`r`nc=3`r`n`r`n[Worker #2]`r`n"
$r3 = Invoke-Task -name 'auto' -task $task -workerSpec 'auto' -p95Todo $busy1
Check "auto delivered the line"             ($r3.add -contains $task)
Check "auto picked worker=2"                ((Get-Notice $r3.out 'ok') -match 'p95_add: ok worker=2 ')
Check "the line is in the [Worker #2] part" ((Get-Index $r3.add '[Worker #2]') -ge 0)

Write-Host "[3b] p95_add_workers = auto without NumWorkers (must warn + deliver)"
$r3b = Invoke-Task -name 'autoNoPrime' -task $task -workerSpec 'auto' -p95Todo $busy1 -primeTxt "# no NumWorkers here`r`n"
Check "still delivered"                     ($r3b.add -contains $task)
Check "warned about prime.txt"              ((Get-Notice $r3b.out 'warn') -match 'prime\.txt')
Check "fell back to header-less"            ((Get-Notice $r3b.out 'warn') -match 'worker=0')

# ------------------------------------------------------------------ [4] range ------------
Write-Host "[4] p95_add_workers = 1-2 (worker 1 empty, worker 2 has 3 lines)"
$busy2 = "[Worker #1]`r`n`r`n[Worker #2]`r`nx=1`r`ny=2`r`nz=3`r`n"
$r4 = Invoke-Task -name 'range' -task $task -workerSpec '1-2' -p95Todo $busy2
Check "the range delivered the line"        ($r4.add -contains $task)
Check "the least loaded worker won (1)"     ((Get-Notice $r4.out 'ok') -match 'p95_add: ok worker=1 ')

# ------------------------------------------------------------------ [5] invalid ----------
Write-Host "[5] invalid specs fall back to a header-less append"
$r5a = Invoke-Task -name 'bad2000' -task $task -workerSpec '2000' -p95Todo $todoBoth
$r5b = Invoke-Task -name 'badabc' -task $task -workerSpec 'abc' -p95Todo $todoBoth
Check "2000 delivered"                      ($r5a.add -contains $task)
Check "2000 warned"                         ($null -ne (Get-Notice $r5a.out 'warn'))
Check "2000 used worker=0"                  ((Get-Notice $r5a.out 'warn') -match 'worker=0')
Check "abc delivered"                       ($r5b.add -contains $task)
Check "abc warned"                          ($null -ne (Get-Notice $r5b.out 'warn'))
Check "abc used worker=0"                   ((Get-Notice $r5b.out 'warn') -match 'worker=0')

# ------------------------------------------------------------------ [6] verbatim ---------
Write-Host "[6] AID + known factors survive byte for byte"
# 1943118631 really divides 2^677-1 (docs/DEV_ECM_WORKTODO.md), so N_eff is legitimate.
$verbatim = 'ECMSTAGE2=ABCDEF0123456789ABCDEF0123456789,1,2,677,-1,"m677_1e3.save",0,0,1,"1943118631"'
$r6 = Invoke-Task -name 'verbatim' -task $verbatim -workerSpec '1' -p95Todo $todoBoth
$hit6 = ($r6.out -match 'FACTOR FOUND')
Write-Host ("      elapsed={0:N1}s  factor found in this run: {1}" -f $r6.seconds, $hit6)
Check "the verbatim line is in worktodo.add"    ($r6.add -contains $verbatim)
Check "the AID is preserved"                    (@($r6.add | Where-Object { $_ -match 'ABCDEF0123456789ABCDEF0123456789' }).Count -eq 1)
Check "the known-factors field is preserved"    (@($r6.add | Where-Object { $_ -match '"1943118631"' }).Count -eq 1)
# Deliberately unconditional: the handoff must not depend on whether a factor was found
# (Prime95's stage 2 still has to run the GCD and report it).
Check "delivered whether or not it hit"         ($r6.add -contains $verbatim)

# ------------------------------------------------------------------ [7] lock -------------
Write-Host "[7] a held lock parks the line, the next task re-delivers it"
$r7a = Invoke-Task -name 'locked' -task $task -workerSpec '1' -p95Todo $todoBoth -lockAgeMinutes 0
Write-Host ("      elapsed={0:N1}s (the lock wait is ~3 s)" -f $r7a.seconds)
Check "the delivery failed loudly"          ($r7a.out -match 'p95_add: pending')
Check "the reason mentions the lock"        ((Get-Notice $r7a.out 'pending') -match 'lock')
Check "it waited for the lock"              ($r7a.seconds -ge 2.5)
Check "no worktodo.add was written"         (-not (Test-Path $r7a.addPath))
Check "the line was parked"                 ((Test-Path $pending) -and ((Get-Content -LiteralPath $pending -Raw) -match [regex]::Escape('m521_1e3.save')))
Check "the pending count is reported"       ((Get-Notice $r7a.out 'pending') -match 'lines=1')
Check "the task still left our queue"       (-not ((Get-Content (Join-Path $r7a.dir 'worktodo.txt') -Raw) -match 'ECMSTAGE2'))

Remove-Item -LiteralPath $r7a.lockPath -Force
$r7b = Invoke-Task -name 'locked' -task $task -workerSpec '1' -p95Todo $todoBoth -KeepAdd
Write-Host ("      second run: {0}" -f (Get-Notice $r7b.out 'ok'))
Check "the next delivery was ok"            ($null -ne (Get-Notice $r7b.out 'ok'))
Check "it reports the re-delivery"          ((Get-Notice $r7b.out 'ok') -match 'pending_delivered=1')
Check "both lines are in worktodo.add"      (@($r7b.add | Where-Object { $_ -match 'm521_1e3\.save' }).Count -eq 2)
Check "the pending file is gone"            (-not (Test-Path $pending))

# ------------------------------------------------------------------ [8] stale lock -------
Write-Host "[8] a lock older than 60 s is preempted instead of waited on"
$r8 = Invoke-Task -name 'stale' -task $task -workerSpec '1' -p95Todo $todoBoth -lockAgeMinutes 5
Write-Host ("      elapsed={0:N1}s" -f $r8.seconds)
Check "the stale lock was preempted"        ($null -ne (Get-Notice $r8.out 'ok'))
Check "delivery was fast (no 3 s wait)"     ($r8.seconds -lt 2.5)
Check "the line is in worktodo.add"         ($r8.add -contains $task)
Check "the stale lock file is gone"         (-not (Test-Path $r8.lockPath))

# ------------------------------------------------------------------ cleanup -------------
Remove-Item -LiteralPath $pending -Force -ErrorAction SilentlyContinue
Check "no pending file is left in the build dir" (-not (Test-Path $pending))

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
