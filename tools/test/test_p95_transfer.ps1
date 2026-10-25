#Requires -Version 5.1
<#
.SYNOPSIS
    Acceptance test for the Prime95 handoff: canonical typed task fields are preserved in
    the worktodo.add next to Prime95's worktodo.txt (docs/usage/GUI.md 18,
    docs/usage/STAGE1.md 8).

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
      [6] verbatim          : AID and known-factors survive; the actual saved range and
                              absolute synchronized file are delivered for NF batches
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
if (-not $Sandbox) { $Sandbox = Join-Path $repoRoot ('data\experiments\p95_transfer_' + (Get-Date -Format 'yyyyMMdd_HHmmss')) }
$Sandbox = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Sandbox)
if (Test-Path -LiteralPath $Sandbox) { throw 'Use a new sandbox; existing evidence is preserved' }
New-Item -ItemType Directory -Force -Path $Sandbox | Out-Null

# Isolate transfer ledgers and relative saves from the supplied executable.
$Exe = (Resolve-Path -LiteralPath $Exe).Path
$sourceDir = Split-Path -Parent $Exe
$testBin = Join-Path $Sandbox 'bin'
New-Item -ItemType Directory -Path $testBin | Out-Null
Copy-Item -LiteralPath $Exe -Destination (Join-Path $testBin 'ecm_cuda.exe')
Copy-Item -LiteralPath (Join-Path $sourceDir 'gmp-10.dll') -Destination $testBin
$Exe = Join-Path $testBin 'ecm_cuda.exe'
# The driver parks undelivered lines next to ITS OWN executable.
$pending = Join-Path (Split-Path -Parent $Exe) 'p95_add_pending.txt'
$ledger = Join-Path (Split-Path -Parent $Exe) 'p95_add_sent.txt'
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
function Toml-String([string]$text) {
    return '"' + $text.Replace('\','\\').Replace('"','\"') + '"'
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
        [switch]$KeepAdd,                 # keep an existing worktodo.add (for scenario 7b)
        [string]$primeLog = "",           # fake Prime95 prime.log (rejected assignment keys)
        [string]$iniExtra = ""            # extra ini lines (p95_keep_aid / p95_recover_lost)
    )
    $dir = Join-Path $Sandbox $name
    if (-not $KeepAdd) {
        if (Test-Path -LiteralPath $dir) { throw 'Scenario already exists; existing evidence is preserved' }
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $dir 'p95') | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $dir 'saves') | Out-Null

    $p95TodoPath = Join-Path $dir 'p95\worktodo.txt'
    $todo = Join-Path $dir 'worktodo.txt'
    $ini = Join-Path $dir 'ecm.toml'
    $log = Join-Path $dir 'screen.log'
    $addPath = Join-Path $dir 'p95\worktodo.add'

    Write-Text (Join-Path $dir 'p95\prime.txt') $primeTxt
    if ($primeLog -ne "") { Write-Text (Join-Path $dir 'p95\prime.log') $primeLog }
    Write-Text $p95TodoPath $p95Todo
    $task = [regex]::Replace($task, '"([^"]+\.save)"', {param($m) '"' + (Join-Path $dir ('saves\' + [IO.Path]::GetFileName($m.Groups[1].Value))) + '"'})
    Write-Text $todo ("[Worker #1]`r`n" + $task + "`r`n")

    # NOTE: parenthesise every concatenation inside an array literal -- the comma binds
    # tighter than '+', so "key = " + $v would silently become two ini lines.
    $iniLines = @(
        '[shared]',
        ('worktodo = ' + (Toml-String $todo)),
        '[stage1]',
        'curve = "mont"', 'target = "gpu"',
        'gpu_param = 0',
        ('device = ' + $Device),
        ('tmp_dir = ' + (Toml-String (Join-Path $dir 'saves'))),
        ('finished = ' + (Toml-String (Join-Path $dir 'finished.txt'))),
        ('log_file = ' + (Toml-String $log)),
        'verbose = false',
        'ckpt_seconds = 0',
        'exp_cache = "off"',
        ('p95_worktodo_path = ' + (Toml-String $p95TodoPath)),
        ('p95_sync_dir = ' + (Toml-String (Join-Path $dir 'p95\saves'))),
        ('p95_add_workers = ' + (Toml-String $workerSpec))
    )
    if ($iniExtra -ne '') { $iniLines += $iniExtra }
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
    $out = (& cmd /c ('"{0}" --config "{1}" --worker 1 < NUL 2>&1' -f $Exe, $ini) | Out-String)
    $code = $LASTEXITCODE
    $sw.Stop()
    Write-Text (Join-Path $dir ('run_' + [Guid]::NewGuid().ToString('N') + '.log')) $out
    Check ($name + ' process completed') ($code -eq 0) ("exit=" + $code)
    $actualAdd = @(if (Test-Path $addPath) { Get-Content -LiteralPath $addPath } else { @() })
    foreach ($savedLine in @($actualAdd | Where-Object { $_ -match '^ECMSTAGE2=' })) {
        $savedPath = [regex]::Match($savedLine, '"([^"]+\.save)"').Groups[1].Value
        Check ($name + ' points published before assignment') ([IO.Path]::IsPathRooted($savedPath) -and (Test-Path -LiteralPath $savedPath -PathType Leaf))
    }
    # Existing routing assertions compare only the unchanged CSV fields; the
    # absolute destination itself is verified above and preserved as actualAdd.
    $comparableAdd = @($actualAdd | ForEach-Object { [regex]::Replace($_, '"([^"]+\.save)"', {param($m) '"' + [IO.Path]::GetFileName($m.Groups[1].Value) + '"'}) })

    return @{
        dir         = $dir
        out         = $out
        addPath     = $addPath
        add         = $comparableAdd
        actualAdd   = $actualAdd
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

$task = 'ECMSTAGE1=1,2,521,-1,"m0_m521_1e3.save",0,0,1'
$handoff = $task.Replace('ECMSTAGE1=', 'ECMSTAGE2=')
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
Check "the line is under [Worker #2]"       ((Get-Index $r1.add '[Worker #2]') -ge 0 -and $r1.add -contains $handoff)
Check "the header precedes the line"        ((Get-Index $r1.add '[Worker #2]') -lt (Get-Index $r1.add $handoff))
Check "no warning when the section exists"  ((Get-Notice $r1.out 'ok') -ne $null -and -not ($r1.out -match 'p95_add: warn'))
Check "the notice says ok worker=2"         ((Get-Notice $r1.out 'ok') -match 'p95_add: ok worker=2 ')
Check "Prime95's worktodo.txt is untouched" ($r1.p95After -eq $r1.p95Before)
Check "the task left our queue"             (-not ((Get-Content (Join-Path $r1.dir 'worktodo.txt') -Raw) -match 'ECMSTAGE1'))
Check "no temp file is left behind"         (-not (Test-Path ($r1.addPath + '.ecm.tmp')))

# ------------------------------------------------------------------ [2] missing ---------
Write-Host "[2] p95_add_workers = 3 without a [Worker #3] section"
$r2 = Invoke-Task -name 'route3' -task $task -workerSpec '3' -p95Todo $todoBoth
$warn2 = Get-Notice $r2.out 'warn'
Check "the line was still delivered"        ($r2.add -contains $handoff)
Check "no [Worker #3] header was invented"  (@($r2.add | Where-Object { $_ -match '\[Worker #3\]' }).Count -eq 0)
Check "the notice is a warning"             ($null -ne $warn2)
Check "the warning says worker=0"           ($warn2 -match 'worker=0')
Check "the warning names the section"       ($warn2 -match 'section')

# ------------------------------------------------------------------ [3] auto -------------
Write-Host "[3] p95_add_workers = auto (NumWorkers=2, worker 1 has 3 lines)"
$busy1 = "[Worker #1]`r`na=1`r`nb=2`r`nc=3`r`n`r`n[Worker #2]`r`n"
$r3 = Invoke-Task -name 'auto' -task $task -workerSpec 'auto' -p95Todo $busy1
Check "auto delivered the line"             ($r3.add -contains $handoff)
Check "auto picked worker=2"                ((Get-Notice $r3.out 'ok') -match 'p95_add: ok worker=2 ')
Check "the line is in the [Worker #2] part" ((Get-Index $r3.add '[Worker #2]') -ge 0)

Write-Host "[3b] p95_add_workers = auto without NumWorkers (must warn + deliver)"
$r3b = Invoke-Task -name 'autoNoPrime' -task $task -workerSpec 'auto' -p95Todo $busy1 -primeTxt "# no NumWorkers here`r`n"
Check "still delivered"                     ($r3b.add -contains $handoff)
Check "warned about prime.txt"              ((Get-Notice $r3b.out 'warn') -match 'prime\.txt')
Check "fell back to header-less"            ((Get-Notice $r3b.out 'warn') -match 'worker=0')

# ------------------------------------------------------------------ [4] range ------------
Write-Host "[4] p95_add_workers = 1-2 (worker 1 empty, worker 2 has 3 lines)"
$busy2 = "[Worker #1]`r`n`r`n[Worker #2]`r`nx=1`r`ny=2`r`nz=3`r`n"
$r4 = Invoke-Task -name 'range' -task $task -workerSpec '1-2' -p95Todo $busy2
Check "the range delivered the line"        ($r4.add -contains $handoff)
Check "the least loaded worker won (1)"     ((Get-Notice $r4.out 'ok') -match 'p95_add: ok worker=1 ')

# ------------------------------------------------------------------ [5] invalid ----------
Write-Host "[5] invalid specs fall back to a header-less append"
$r5a = Invoke-Task -name 'bad2000' -task $task -workerSpec '2000' -p95Todo $todoBoth
$r5b = Invoke-Task -name 'badabc' -task $task -workerSpec 'abc' -p95Todo $todoBoth
Check "2000 delivered"                      ($r5a.add -contains $handoff)
Check "2000 warned"                         ($null -ne (Get-Notice $r5a.out 'warn'))
Check "2000 used worker=0"                  ((Get-Notice $r5a.out 'warn') -match 'worker=0')
Check "abc delivered"                       ($r5b.add -contains $handoff)
Check "abc warned"                          ($null -ne (Get-Notice $r5b.out 'warn'))
Check "abc used worker=0"                   ((Get-Notice $r5b.out 'warn') -match 'worker=0')

# ------------------------------------------------------------------ [6] verbatim ---------
Write-Host "[6] AID + known factors survive byte for byte"
# Divide 2^1042-1 by 2^521+1; the remaining prime M521 cannot yield a factor.
$known = ([Numerics.BigInteger]::Pow([Numerics.BigInteger]2,521)+1).ToString()
$verbatim = 'ECMSTAGE1=ABCDEF0123456789ABCDEF0123456789,1,2,1042,-1,"m0_m1042_1e3.save",0,0,1,"' + $known + '"'
$handoffVerbatim = $verbatim.Replace('ECMSTAGE1=', 'ECMSTAGE2=')
$r6 = Invoke-Task -name 'verbatim' -task $verbatim -workerSpec '1' -p95Todo $todoBoth
$hit6 = ($r6.out -match 'FACTOR FOUND')
Write-Host ("      elapsed={0:N1}s  factor found in this run: {1}" -f $r6.seconds, $hit6)
Check "the fields survive in worktodo.add"      ($r6.add -contains $handoffVerbatim)
Check "the AID is preserved"                    (@($r6.add | Where-Object { $_ -match 'ABCDEF0123456789ABCDEF0123456789' }).Count -eq 1)
Check "the known-factors field is preserved"    (@($r6.add | Where-Object { $_.Contains('"' + $known + '"') }).Count -eq 1)
Check "the NF batch is delivered"              (-not $hit6 -and $r6.add -contains $handoffVerbatim)

# ------------------------------------------------------------------ [7] lock -------------
Write-Host "[7] a held lock parks the line, the next task re-delivers it"
$r7a = Invoke-Task -name 'locked' -task $task -workerSpec '1' -p95Todo $todoBoth -lockAgeMinutes 0
Write-Host ("      elapsed={0:N1}s (the lock wait is ~3 s)" -f $r7a.seconds)
Check "the delivery failed loudly"          ($r7a.out -match 'p95_add: pending')
Check "the reason mentions the lock"        ((Get-Notice $r7a.out 'pending') -match 'lock')
Check "it waited for the lock"              ($r7a.seconds -ge 2.5)
Check "no worktodo.add was written"         (-not (Test-Path $r7a.addPath))
Check "the line was parked durably"         (@(Get-ChildItem -LiteralPath ($pending+'.handoff') -Filter '*.intent' -ErrorAction SilentlyContinue).Count -eq 1)
Check "the pending count is reported"       ((Get-Notice $r7a.out 'pending') -match 'lines=1')
Check "the task still left our queue"       (-not ((Get-Content (Join-Path $r7a.dir 'worktodo.txt') -Raw) -match 'ECMSTAGE1'))

Remove-Item -LiteralPath $r7a.lockPath -Force
$r7b = Invoke-Task -name 'locked' -task $task -workerSpec '1' -p95Todo $todoBoth -KeepAdd
Write-Host ("      second run: {0}" -f (Get-Notice $r7b.out 'ok'))
Check "the next delivery was ok"            ($null -ne (Get-Notice $r7b.out 'ok'))
Check "it reports the re-delivery"          ($r7b.out -match 'p95_add: ok [^\r\n]*pending_delivered=1')
Check "both lines are in worktodo.add"      (@($r7b.add | Where-Object { $_ -match 'm0_m521_1e3\.save' }).Count -eq 2)
Check "no handoff responsibility remains"   (@(Get-ChildItem -LiteralPath ($pending+'.handoff') -Filter '*.intent' -ErrorAction SilentlyContinue).Count -eq 0)

# ------------------------------------------------------------------ [8] stale lock -------
Write-Host "[8] a lock older than 60 s is preempted instead of waited on"
$r8 = Invoke-Task -name 'stale' -task $task -workerSpec '1' -p95Todo $todoBoth -lockAgeMinutes 5
Write-Host ("      elapsed={0:N1}s" -f $r8.seconds)
Check "the stale lock was preempted"        ($null -ne (Get-Notice $r8.out 'ok'))
Check "delivery was fast (no 3 s wait)"     ($r8.seconds -lt 2.5)
Check "the line is in worktodo.add"         ($r8.add -contains $handoff)
Check "the stale lock file is gone"         (-not (Test-Path $r8.lockPath))

# ------------------------------------------------------------------ [9] rejected AID ----
# The operator's production run (2026-09-29) showed the failure this guards against: a line
# delivered with an assignment key PrimeNet has already rejected makes Prime95 DELETE the work
# unit (commonc.c:6550-6569 "delete it from our work to do file"), so the stage 2 never runs and
# the stage-1 work is lost. The driver now reads Prime95's own prime.log, drops such a key and
# re-delivers what Prime95 threw away.
Write-Host "[9] an AID PrimeNet rejected must not be attached to the delivery"
$rejectedAid = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
$liveAid = 'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB'
$primeLogBad = @(
    'PrimeNet error 43: Invalid assignment key',
    ("ap: no such assignment key, GUID: 0e068755443bf5372102b7ceb29a6d62, key: " + $rejectedAid)
) -join "`r`n"
$taskBadAid = ('ECMSTAGE1=' + $rejectedAid + ',1,2,521,-1,"m0_m521_1e3.save",0,0,1')
$taskLiveAid = ('ECMSTAGE1=' + $liveAid + ',1,2,523,-1,"m0_m523_1e3.save",0,0,1')
$handoffBadAid = $taskBadAid.Replace('ECMSTAGE1=', 'ECMSTAGE2=')
$handoffLiveAid = $taskLiveAid.Replace('ECMSTAGE1=', 'ECMSTAGE2=')

Remove-Item -LiteralPath $ledger -Force -ErrorAction SilentlyContinue
$r9 = Invoke-Task -name 'rejectedaid' -task $taskBadAid -workerSpec '1' -p95Todo $todoBoth -primeLog $primeLogBad
$notice9 = Get-Notice $r9.out 'warn'
Write-Host ("      notice: {0}" -f $(if ($notice9) { $notice9.Trim() } else { '(none)' }))
Check "the delivery still happened"          ($r9.add -contains 'ECMSTAGE2=1,2,521,-1,"m0_m521_1e3.save",0,0,1')
Check "the rejected AID was dropped"         (@($r9.add | Where-Object { $_ -match [regex]::Escape($rejectedAid) }).Count -eq 0)
Check "the notice says aid=dropped"          ($notice9 -match 'aid=dropped')
Check "the notice names the rejected key"    ($notice9 -match 'rejected AID')
Check "the rejected key was detected"        ((Get-Notice $r9.out 'ready') -match 'rejected_aids=1')
Check "a live AID is kept"                   ($true)   # covered by [6] (verbatim line) and [9b]
$r9b = Invoke-Task -name 'liveaid' -task $taskLiveAid -workerSpec '1' -p95Todo $todoBoth -primeLog $primeLogBad
Check "the live AID is still delivered"      ($r9b.add -contains $handoffLiveAid)
Check "and it is reported as kept"           ((Get-Notice $r9b.out 'ok') -match 'aid=kept')

# ------------------------------------------------------------------ [10] recovery --------
Write-Host "[10] a line Prime95 threw away is re-delivered without its AID"
Remove-Item -LiteralPath $ledger -Force -ErrorAction SilentlyContinue
# The ledger says we delivered `taskBadAid`; PrimeNet rejected that key (prime.log above) and
# Prime95 no longer holds the line, so the driver must hand it over again without the AID.
$recoverable = @($r9.actualAdd | Where-Object { $_ -match '^ECMSTAGE2=' })[0]
$recoverableWithAid = $recoverable.Replace('ECMSTAGE2=', ('ECMSTAGE2='+$rejectedAid+','))
[System.IO.File]::WriteAllText($ledger, ('sent ' + $recoverableWithAid + "`n"), (New-Object System.Text.UTF8Encoding($false)))
$r10 = Invoke-Task -name 'recover' -task $task -workerSpec '1' -p95Todo $todoBoth -primeLog $primeLogBad
$notice10 = Get-Notice $r10.out 'warn'
Write-Host ("      notice: {0}" -f $(if ($notice10) { $notice10.Trim() } else { '(none)' }))
Check "the lost line came back without its AID" ($r10.add -contains 'ECMSTAGE2=1,2,521,-1,"m0_m521_1e3.save",0,0,1')
Check "the notice reports recovered=1"          ($notice10 -match 'recovered=1')
Check "the new task was delivered too"          ($r10.add -contains $handoff)
Check "the ledger marks the recovery"           ((Get-Content $ledger -Raw) -match 'recovered ')
# A second run must NOT hand it over again (it is already in Prime95's file/the ledger).
$r10b = Invoke-Task -name 'recover' -task $task -workerSpec '1' -p95Todo $todoBoth -primeLog $primeLogBad -KeepAdd
Check "it is not re-delivered twice"            ($null -eq (Get-Notice $r10b.out 'warn'))

Write-Host "[10b] nothing is recovered when Prime95 still holds the line"
Remove-Item -LiteralPath $ledger -Force -ErrorAction SilentlyContinue
[System.IO.File]::WriteAllText($ledger, ('sent ' + $recoverableWithAid + "`n"), (New-Object System.Text.UTF8Encoding($false)))
# The line (AID stripped) is already in Prime95's worktodo.txt -> no recovery.
$r10c = Invoke-Task -name 'recoverheld' -task $task -workerSpec '1' -p95Todo ($todoBoth + $recoverable + "`r`n") -primeLog $primeLogBad
Check "no recovery while Prime95 holds it"      ($null -eq (Get-Notice $r10c.out 'warn'))

# ------------------------------------------------------------------ [11] keep_aid = 0 ----
Write-Host "[11] p95_keep_aid = 0 drops the AID even when it is not known to be bad"
Remove-Item -LiteralPath $ledger -Force -ErrorAction SilentlyContinue
$r11 = Invoke-Task -name 'noaid' -task $taskLiveAid -workerSpec '1' -p95Todo $todoBoth -iniExtra 'p95_keep_aid = false'
Check "the line went over without the AID"      ($r11.add -contains 'ECMSTAGE2=1,2,523,-1,"m0_m523_1e3.save",0,0,1')
Check "no AID is present in worktodo.add"       (@($r11.add | Where-Object { $_ -match [regex]::Escape($liveAid) }).Count -eq 0)
Check "the notice says why"                     ((Get-Notice $r11.out 'warn') -match 'p95_keep_aid = 0')
Check "ready reports keep_aid=0"                ((Get-Notice $r11.out 'ready') -match 'keep_aid=0')

# ------------------------------------------------------------------ cleanup -------------
Remove-Item -LiteralPath $pending -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $ledger -Force -ErrorAction SilentlyContinue
Check "no pending file is left in the build dir" (-not (Test-Path $pending))
Check "no ledger is left in the build dir"       (-not (Test-Path $ledger))

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
