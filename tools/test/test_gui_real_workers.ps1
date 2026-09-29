#Requires -Version 5.1
<#
.SYNOPSIS
    M2 acceptance with the REAL driver and REAL GPUs: two workers, one per device,
    each consuming only its own [Worker #N] section of one worktodo file.

.DESCRIPTION
    Builds a sandbox where

        ecm.ini       : global keys + [GUI] (exe = the real ecm_cuda.exe) + [Worker #1]
                        (device 0) + [Worker #2] (device 1)
        worktodo.txt  : one ECMSTAGE2= line under [Worker #1] and one under [Worker #2]

    then starts ecm_gui, lets it autostart both workers, and waits (bounded) for both
    to finish. The task is tiny on purpose (M991, B1=1e4, 8 curves -> a few seconds),
    so this is safe to run on a machine that is doing other work, and it proves:

      * the GUI spawns the real driver with --worker N (its own ini/worktodo section);
      * each worker consumes ONLY its own section (both lines disappear from the SAME
        file, and neither worker ran the other's task);
      * per-worker logs (screen_1.log / screen_2.log) and the shared finished file;
      * closing the GUI leaves no process behind.

.PARAMETER Device2
    CUDA device index for Worker #2 (default 1). Pass the same index as Device1 to
    run both on one card when the machine has a single GPU.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_gui_real_workers.ps1
#>
param(
    [string]$Exe = "",
    [string]$EcmCuda = "",
    [string]$Sandbox = "",
    [int]$Device1 = 0,
    [int]$Device2 = 1,
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
if (-not $Sandbox) { $Sandbox = Join-Path $PSScriptRoot '_run\gui_real_workers' }
if (Test-Path $Sandbox) { Remove-Item $Sandbox -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $Sandbox 'saves') | Out-Null

$script:pass = 0
$script:fail = 0
# The GUI keeps the trace file open while it runs: read it with shared access.
function Read-TextShared([string]$path) {
    if (-not (Test-Path $path)) { return "" }
    try {
        $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Open,
                                     [System.IO.FileAccess]::Read,
                                     [System.IO.FileShare]::ReadWrite)
        $sr = New-Object System.IO.StreamReader($fs)
        $text = $sr.ReadToEnd()
        $sr.Close(); $fs.Close()
        return $text
    } catch { return "" }
}
function Check([string]$name, [bool]$ok, [string]$detail = "") {
    if ($ok) { $script:pass++; Write-Host ("  [ok]   " + $name) }
    else {
        $script:fail++
        Write-Host ("  [FAIL] " + $name + $(if ($detail) { " -- " + $detail } else { "" })) -ForegroundColor Red
    }
}

if (-not ("Win32GuiReal" -as [type])) {
    Add-Type @'
using System; using System.Runtime.InteropServices; using System.Text;
public static class Win32GuiReal {
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

$iniPath = Join-Path $Sandbox 'ecm.ini'
$todo = Join-Path $Sandbox 'worktodo.txt'
$finished = Join-Path $Sandbox 'finished.txt'
$enc = New-Object System.Text.UTF8Encoding($false)

# M991/M997 with B1=1e4 and 8 curves: seconds per worker. The save names must end with
# the B1 token ("m{n}_{b1}.save") -- the driver extracts B1 from the LAST "_" token, so
# a suffix like "_w1" makes it reject the line ("# ERROR cannot extract B1"). Distinct
# exponents keep the two workers off each other's .save file.
$task1 = 'ECMSTAGE2=1,2,991,-1,"m991_1e4.save",0,0,8'
$task2 = 'ECMSTAGE2=1,2,997,-1,"m997_1e4.save",0,0,8'
[System.IO.File]::WriteAllText($todo, (@(
    '# two sections, one task each',
    '[Worker #1]',
    $task1,
    '',
    '[Worker #2]',
    $task2) -join "`r`n") + "`r`n", $enc)

# NOTE: inside an array literal the comma binds tighter than '+', so every
# concatenation MUST be parenthesised -- otherwise "exe = <path>" becomes two ini
# lines and the GUI falls back to a PATH lookup (this cost one debugging round).
$saves = Join-Path $Sandbox 'saves'
$iniLines = @(
    '# M2 real-driver sandbox',
    'method = gpu',
    'backend = auto',
    'gpucurves = 8',
    ('tmp_dir = ' + $saves),
    ('finished = ' + $finished),
    ('worktodo = ' + $todo),
    'verbose = 0',
    'ckpt_seconds = 0',
    'save_sync_dir_1 =',
    'save_sync_dir_2 =',
    '',
    '[GUI]',
    'NumWorkers = 2',
    ('exe = ' + $EcmCuda),
    'language = english',
    'refresh_hz = 10',
    '',
    '[Worker #1]',
    ('name = gpu' + $Device1),
    ('log_file = ' + (Join-Path $Sandbox 'screen_1.log')),
    ('device = ' + $Device1),
    'autostart = 1',
    '',
    '[Worker #2]',
    ('name = gpu' + $Device2),
    ('log_file = ' + (Join-Path $Sandbox 'screen_2.log')),
    ('device = ' + $Device2),
    'autostart = 1'
)
[System.IO.File]::WriteAllText($iniPath, (($iniLines -join "`r`n") + "`r`n"), $enc)

$trace = Join-Path (Split-Path -Parent $Exe) 'ecm_gui_trace.log'
Remove-Item $trace -ErrorAction SilentlyContinue

Write-Host ("exe      : " + $Exe)
Write-Host ("ecm_cuda : " + $EcmCuda)
Write-Host ("sandbox  : " + $Sandbox)
Write-Host ("tasks    : M991 B1=1e4, 8 curves, device {0} + device {1}" -f $Device1, $Device2)
Write-Host ""
Write-Host "[1] start the GUI; both workers autostart"

$proc = Start-Process -FilePath $Exe -ArgumentList @('-ini', $iniPath, '--trace') -PassThru
$hwnd = [IntPtr]::Zero
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline -and $hwnd -eq [IntPtr]::Zero) {
    Start-Sleep -Milliseconds 400
    $proc.Refresh()
    if ($proc.HasExited) { break }
    $hwnd = [Win32GuiReal]::Find([uint32]$proc.Id, 'ecm_gui')
}
Check "GUI window appeared" ($hwnd -ne [IntPtr]::Zero)

Write-Host ("[2] wait up to {0}s for both workers to finish their task" -f $TimeoutSeconds)
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$done1 = $false; $done2 = $false
while ($sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
    Start-Sleep -Seconds 2
    if (-not (Test-Path $trace)) { continue }
    $log = Read-TextShared $trace
    $done1 = $log -match 'worker 1: state QueueEmpty'
    $done2 = $log -match 'worker 2: state QueueEmpty'
    if ($done1 -and $done2) { break }
    $proc.Refresh()
    if ($proc.HasExited) { break }
}
$log = Read-TextShared $trace
Check "worker 1 finished (QueueEmpty)" $done1 (("elapsed {0:N0}s" -f $sw.Elapsed.TotalSeconds))
Check "worker 2 finished (QueueEmpty)" $done2 (("elapsed {0:N0}s" -f $sw.Elapsed.TotalSeconds))
Check "worker 1 reached Running" ($log -match 'worker 1: state Running')
Check "worker 2 reached Running" ($log -match 'worker 2: state Running')
Check "no restart was needed" (-not ($log -match 'restart #'))

Write-Host "[3] close the GUI"
if ($hwnd -ne [IntPtr]::Zero) { [Win32GuiReal]::Close($hwnd) }
if (-not $proc.WaitForExit(30000)) { $proc.Kill(); Check "GUI exited" $false "had to kill it" }
else { Check "GUI exited cleanly" ($proc.ExitCode -eq 0) ("exit code " + $proc.ExitCode) }

Write-Host "[4] each worker consumed only its own section"
$after = [System.IO.File]::ReadAllText($todo)
Check "worker 1's line is gone"  (-not ($after -match [regex]::Escape($task1)))
Check "worker 2's line is gone"  (-not ($after -match [regex]::Escape($task2)))
Check "no line was marked as ERROR" (-not ($after -match '# ERROR'))
Check "both section headers survive" (($after -match '\[Worker #1\]') -and ($after -match '\[Worker #2\]'))
$fin = if (Test-Path $finished) { [System.IO.File]::ReadAllText($finished) } else { '' }
Check "the shared finished file has both tasks" (([regex]::Matches($fin, 'ECMSTAGE2=')).Count -eq 2) $fin

Write-Host "[5] per-worker logs prove each worker ran ITS OWN task"
# NOTE: the GPU stage-1 path writes .save files next to the ecm_cuda executable (not
# into tmp_dir), and a hit legitimately produces no save at all -- so the durable
# per-worker evidence here is the log file each worker was told to write.
$log1 = if (Test-Path (Join-Path $Sandbox 'screen_1.log')) { [System.IO.File]::ReadAllText((Join-Path $Sandbox 'screen_1.log')) } else { '' }
$log2 = if (Test-Path (Join-Path $Sandbox 'screen_2.log')) { [System.IO.File]::ReadAllText((Join-Path $Sandbox 'screen_2.log')) } else { '' }
Check "worker 1 wrote its own log" ($log1.Length -gt 0)
Check "worker 2 wrote its own log" ($log2.Length -gt 0)
Check "worker 1 log shows the M991 task" ($log1 -match 'START: ECMSTAGE2=1,2,991,')
Check "worker 2 log shows the M997 task" ($log2 -match 'START: ECMSTAGE2=1,2,997,')
Check "worker 1 log never mentions M997" (-not ($log1 -match '2,997,'))
Check "worker 2 log never mentions M991" (-not ($log2 -match '2,991,'))
Check "worker 1 log ends with queue done" ($log1 -match 'queue done, 1 task')
Check "worker 2 log ends with queue done" ($log2 -match 'queue done, 1 task')
Check "worker 1 used device 0" ($log1 -match 'device 0')
Check "worker 2 used device 1" ($log2 -match 'device 1')
Check "no stray ecm_cuda processes" (@(Get-Process -Name 'ecm_cuda' -ErrorAction SilentlyContinue).Count -eq 0)

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
