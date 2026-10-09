#Requires -Version 5.1
<#
.SYNOPSIS
    Acceptance test for the progress cadence: pipe every line, file every N seconds
    (ini key progress_log_seconds), and the 100% line always written
    (docs/usage/GUI.md 7.2).

.DESCRIPTION
    One real queue-mode task (M521, B1=1e5, 4 curves, param0, device 0) is run four times
    with a different progress_log_seconds, and stdout is compared against the log FILE:

      default (key absent) : the file gets ONE progress line -- the 100% one -- while the
                             pipe gets dozens (the GUI tails stdout and needs ~200 ms)
      0                    : the file gets only the 100% line, and still gets every
                             non-progress line (the task is not invisible in the log)
      1                    : about one line per second in the file, far fewer than stdout
      -1                   : every progress line reaches the file again (old behaviour)

    M521 is a Mersenne PRIME exponent, so no factor can be found: every run covers the
    whole scalar and the line counts are not luck-dependent.
    ASCII only on purpose (Windows PowerShell 5.1 needs a BOM for non-ASCII .ps1).

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_progress_cadence.ps1
#>
param(
    [string]$Exe = "",
    [string]$Sandbox = "",
    [int]$Device = 0,
    [string]$B1 = "1e5",
    [int]$Curves = 4
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Exe) {
    foreach ($cand in @("$repoRoot\build_cuda_cmake\ecm_cuda.exe",
                        "$repoRoot\build_vs18\Release\ecm_cuda.exe",
                        "$repoRoot\build_cuda_dev\ecm_cuda.exe")) {
        if (Test-Path $cand) { $Exe = $cand; break }
    }
}
if (-not $Exe -or -not (Test-Path $Exe)) {
    Write-Host "FAIL: no ecm_cuda executable found (pass -Exe <path>)" -ForegroundColor Red
    exit 2
}
if (-not $Sandbox) { $Sandbox = Join-Path $PSScriptRoot '_run\progress_cadence' }
if (Test-Path $Sandbox) { Remove-Item $Sandbox -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $Sandbox 'saves') | Out-Null

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
$todo = Join-Path $Sandbox 'worktodo.txt'
$ini = Join-Path $Sandbox 'ecm.ini'
$finished = Join-Path $Sandbox 'finished.txt'
$saves = Join-Path $Sandbox 'saves'
$logFile = Join-Path $Sandbox 'screen.log'
$task = 'ECMSTAGE2=1,2,521,-1,"m521_{0}.save",0,0,{1}' -f $B1, $Curves

# One run: fresh worktodo + log, one ini line for the cadence (or none for "default").
# Returns @{ out; log; seconds }.
function Invoke-Cadence([string]$seconds) {
    # The log must REALLY be gone before the run: a delete that fails (the file is held open
    # by a leftover driver) leaves the previous run's lines in place and the counts below
    # would be meaningless. Retry, then say so instead of reporting a cadence bug.
    for ($try = 0; $try -lt 10; $try++) {
        if (-not (Test-Path $logFile)) { break }
        Remove-Item -LiteralPath $logFile -Force -ErrorAction SilentlyContinue
        if (-not (Test-Path $logFile)) { break }
        Start-Sleep -Milliseconds 300
    }
    if (Test-Path $logFile) { throw "cannot clear $logFile (a worker still holds it open)" }
    [System.IO.File]::WriteAllText($todo, ("[Worker #1]`r`n" + $task + "`r`n"), $enc)
    # NOTE: parenthesise every concatenation inside an array literal -- the comma binds
    # tighter than '+', so "key = " + $v would silently become two ini lines.
    $iniLines = @(
        'method = gpu',
        'gpu_param = 0',
        ('gpucurves = ' + $Curves),
        ('tmp_dir = ' + $saves),
        ('finished = ' + $finished),
        ('worktodo = ' + $todo),
        ('device = ' + $Device),
        'verbose = 0',
        'ckpt_seconds = 0',
        ('log_file = ' + $logFile)
    )
    if ($seconds -ne '') { $iniLines += ('progress_log_seconds = ' + $seconds) }
    [System.IO.File]::WriteAllText($ini, (($iniLines -join "`r`n") + "`r`n"), $enc)

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $out = (& cmd /c ('"{0}" -ini "{1}" --worker 1 < NUL 2>&1' -f $Exe, $ini) | Out-String)
    $sw.Stop()
    $log = @(Get-Content -LiteralPath $logFile -ErrorAction SilentlyContinue)
    return @{
        out     = $out
        log     = $log
        seconds = $sw.Elapsed.TotalSeconds
    }
}

function Count-Progress([string[]]$lines) {
    return @($lines | Where-Object { $_ -match '(GPU|stage1): \[' }).Count
}
function Count-Final([string[]]$lines) {
    return @($lines | Where-Object { $_ -match '(GPU|stage1): \[.*\] 100\.0%' }).Count
}

Write-Host ("driver  : " + $Exe)
Write-Host ("task    : " + $task + " (2^521-1, device $Device)")
Write-Host ("sandbox : " + $Sandbox)
Write-Host ""

# ------------------------------------------------------------------ default ---------
Write-Host "[1] no progress_log_seconds (default 60 s)"
$r = Invoke-Cadence ''
$stdoutProgress = Count-Progress @($r.out -split "`r?`n")
$fileProgress = Count-Progress $r.log
$fileProgressLines = @($r.log | Where-Object { $_ -match '(GPU|stage1): \[' })
Write-Host ("      elapsed={0:N1}s  stdout progress={1}  file progress={2}" -f $r.seconds, $stdoutProgress, $fileProgress)
if ($fileProgress -ne 1) {
    # A run always writes the FIRST progress line immediately (the gate starts closed, so
    # the file shows where the task stands) plus every 100% line; anything else means the
    # cadence leaked.
    Write-Host "      file lines:"
    $fileProgressLines | ForEach-Object { Write-Host ("        " + $_) }
}
Check "the task ran (stdout has progress lines)" ($stdoutProgress -ge 10) ("stdout=$stdoutProgress")
Check "the pipe keeps its ~200 ms cadence" ($stdoutProgress / $r.seconds -ge 1.0) ("{0:N2} lines/s" -f ($stdoutProgress / $r.seconds))
Check "the file gets exactly one progress line" ($fileProgress -eq 1) ("file=$fileProgress :: " + ($fileProgressLines -join ' | '))
Check "that one line is the 100% line" (Count-Final $r.log -eq 1)
Check "the log still has the other lines" ((@($r.log | Where-Object { $_ -match 'queue done|task\(s\) processed' }).Count -ge 1))

# ------------------------------------------------------------------ 0 ----------------
Write-Host "[2] progress_log_seconds = 0 (no progress lines in the file)"
$r0 = Invoke-Cadence '0'
Write-Host ("      elapsed={0:N1}s  stdout progress={1}  file progress={2}" -f $r0.seconds, (Count-Progress @($r0.out -split "`r?`n")), (Count-Progress $r0.log))
Check "stdout is unaffected"        ((Count-Progress @($r0.out -split "`r?`n")) -ge 10)
Check "file progress == 1 (100% only)" ((Count-Progress $r0.log) -eq 1) ("file=$(Count-Progress $r0.log)")
Check "the 100% line is the last progress line" `
    ((@($r0.log | Where-Object { $_ -match '(GPU|stage1): \[' })[-1]) -match '100\.0%')
Check "non-progress lines are still logged" ((@($r0.log | Where-Object { $_ -match 'queue done|task\(s\) processed' }).Count -ge 1))
Check "the 100% line really reports 100.0%"  ((Count-Final $r0.log) -eq 1)

# ------------------------------------------------------------------ 1 ----------------
Write-Host "[3] progress_log_seconds = 1 (about one line per second)"
$r1 = Invoke-Cadence '1'
$sp1 = Count-Progress @($r1.out -split "`r?`n")
$fp1 = Count-Progress $r1.log
Write-Host ("      elapsed={0:N1}s  stdout progress={1}  file progress={2}" -f $r1.seconds, $sp1, $fp1)
Check "the file is rate-limited"        ($fp1 -lt $sp1) ("file=$fp1 stdout=$sp1")
Check "about one line per second"       ($fp1 -ge 2 -and $fp1 -le ([Math]::Ceiling($r1.seconds) + 2)) `
    ("file=$fp1 elapsed={0:N1}s" -f $r1.seconds)
Check "the file gets intermediate lines too (unlike = 0)" `
    ((@($r1.log | Where-Object { $_ -match '(GPU|stage1): \[' -and $_ -notmatch '100\.0%' }).Count) -ge 1)
Check "the 100% line is written"        ((Count-Final $r1.log) -eq 1)

# ------------------------------------------------------------------ -1 ---------------
Write-Host "[4] progress_log_seconds = -1 (every line, pre-D4 behaviour)"
$r2 = Invoke-Cadence '-1'
$sp2 = Count-Progress @($r2.out -split "`r?`n")
$fp2 = Count-Progress $r2.log
Write-Host ("      elapsed={0:N1}s  stdout progress={1}  file progress={2}" -f $r2.seconds, $sp2, $fp2)
Check "the file gets every progress line" ($fp2 -eq $sp2) ("file=$fp2 stdout=$sp2")
Check "that is more than the 1 s run"     ($fp2 -gt $fp1 -or $sp2 -lt 5)

# ------------------------------------------------------------------ template ---------
Write-Host "[5] the default ini template documents the key"
$fresh = Join-Path $Sandbox 'fresh\ecm.ini'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $fresh) | Out-Null
$null = (& cmd /c ('cd /d "{0}" && "{1}" -ini "{2}" --worker 1 < NUL 2>&1' -f (Split-Path -Parent $fresh), $Exe, $fresh) | Out-String)
$freshText = if (Test-Path $fresh) { Get-Content -LiteralPath $fresh -Raw } else { '' }
Check "the driver created a default ini" ($freshText.Length -gt 100)
Check "it contains progress_log_seconds = 60" ($freshText -match '(?m)^\s*progress_log_seconds\s*=\s*60\s*$')
Check "the key is explained (file vs pipe)"   ($freshText -match 'rate-limited|log_file')

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
