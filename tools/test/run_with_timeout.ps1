#Requires -Version 5.1
<#
.SYNOPSIS
    Run a program with a HARD wall-clock timeout, so a hung child can never block a test for
    tens of minutes.

.DESCRIPTION
    Why this exists: on Windows a program that dies with an access violation can be held open
    by a Windows Error Reporting dialog ("该内存不能为 read").  The dialog keeps the child
    process alive, the parent `cmd.exe` therefore never returns, and whoever launched it --
    a test script, a suite entry, or an agent's command line -- just waits until its own much
    longer timeout fires.  Three defences, and this script is the third:

      1. per machine : WER对话框已關 (HKCU\Software\Microsoft\Windows\Windows Error
                       Reporting\DontShowUI = 1), see docs/DEV_WINDOWS_CRASH_HANDLING.md
      2. per process : our own tools call SetErrorMode(SEM_NOGPFAULTERRORBOX) at the top of
                       main (tools/bench/no_crash_dialog.h)
      3. per run     : THIS SCRIPT -- the child gets a wall-clock budget; on expiry the whole
                       process tree is killed (taskkill /T /F) and a distinct exit code is
                       returned, so a crash-with-dialog is reported as "timeout" instead of
                       hanging.

    Output goes to a FILE (not a pipe), so nothing can block on a full pipe buffer, and the
    child is launched through `cmd /c` so a shell redirect does the writing.

.PARAMETER Exe
    Program to run.

.PARAMETER Arguments
    Argument array for the program.

.PARAMETER TimeoutSec
    Wall-clock budget.  Default 120.  On expiry the process tree is killed.

.PARAMETER Log
    Where the child's stdout+stderr go.  Default: a temp file next to the exe.

.OUTPUTS
    Writes "exit=<code>" (or "exit=timeout") and "log=<path>" to the pipeline and returns the
    child's exit code; 124 is returned when the timeout fired (the same convention as
    coreutils `timeout`), so a caller can tell a crash-with-dialog from a real failure.

.EXAMPLE
    powershell -File tools\test\run_with_timeout.ps1 -Exe build_cuda_cmake\stage2_tree_gpu.exe `
        -Arguments @('--evaluate-batched','--n','12345') -TimeoutSec 300
#>
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    # A STRING, not an array: powershell -File flattens an array into one string anyway,
    # so an array here silently produces 'A positional parameter cannot be found' (measured).
    [string]$Arguments = '',
    [int]$TimeoutSec = 120,
    [string]$Log = ''
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Exe)) { Write-Host "run_with_timeout: not found: $Exe" -ForegroundColor Red; exit 2 }
if (-not $Log) {
    $dir = Split-Path -Parent $Exe
    if (-not $dir) { $dir = '.' }
    $Log = Join-Path $dir ('_run_' + [System.IO.Path]::GetFileNameWithoutExtension($Exe) + '_' +
                           (Get-Date -Format 'HHmmss') + '.log')
}

# Quote ONLY what contains whitespace, and in cmd's own form.  `cmd /c "<exe>" args` strips
# the outer quotes when the command line STARTS with a quote, which made cmd return 1
# immediately with an empty log (measured); leaving unquoted paths unquoted avoids the whole
# class of problem, and the doubled-quote form is what cmd wants when quoting is needed.
function Q([string]$s) { if ($s -match '\s') { '""' + $s + '""' } else { $s } }
$line = (Q $Exe) + ' ' + $Arguments + ' > ' + (Q $Log) + ' 2>&1'

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$timedOut = $false
$code = -1

# .NET Process with UseShellExecute = $false, NOT Start-Process: Start-Process re-joins
# -ArgumentList into a command line and mangles the quotes inside `$line` (measured: the
# child never started and the log stayed empty).  No redirection here either -- the `> log`
# in $line is done by cmd itself, so no pipe can fill up and block us.
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $env:ComSpec
$psi.Arguments = '/c ' + $line
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$p = [System.Diagnostics.Process]::Start($psi)
if (-not $p.WaitForExit($TimeoutSec * 1000)) {
    $timedOut = $true
    # Kill the CHILDREN first (the program itself), then the cmd.exe launcher: .NET
    # Process.Kill() only kills the direct child (PS 5.1 has no Kill(entireTree)), and
    # taskkill.exe is denied by the harness sandbox ("ERROR: Access denied", measured), so
    # the tree is walked through CIM instead.
    function Stop-Tree([int]$id) {
        try {
            Get-CimInstance Win32_Process -Filter ("ParentProcessId=$id") -ErrorAction SilentlyContinue |
                ForEach-Object { Stop-Tree ([int]$_.ProcessId) }
        } catch { }
        try { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue } catch { }
    }
    # taskkill /T /F is the clean way to kill a tree; it is denied in a confined sandbox
    # ("ERROR: Access denied", measured), in which case the CIM walk is the fallback.
    $killed = $false
    try {
        $null = & taskkill.exe /T /F /PID $p.Id 2>&1
        if ($LASTEXITCODE -eq 0) { $killed = $true }
    } catch { }
    if (-not $killed) { Stop-Tree $p.Id }
    # give the OS a moment, then make sure the grandchild is really gone: a stray child
    # keeps the log file open and makes the NEXT run fail instantly with a stale log.
    Start-Sleep -Milliseconds 400
    if (-not $killed) { Stop-Tree $p.Id }
    Start-Sleep -Milliseconds 200
    try { if (-not $p.HasExited) { $p.Kill() } } catch { }
} else {
    $code = $p.ExitCode
}
$sw.Stop()

$status = if ($timedOut) { 'timeout' } else { "$code" }
Write-Host ("exit=" + $status + "  seconds=" + [math]::Round($sw.Elapsed.TotalSeconds, 1) + "  log=" + $Log)
if (Test-Path $Log) {
    Get-Content $Log -Tail 12 | ForEach-Object { Write-Host ("    " + $_) }
}
if ($timedOut) { exit 124 }
exit $code
