#Requires -Version 5.1
<#
.SYNOPSIS
    Exit flow of ecm_gui: confirmation modal, and a checkpoint written before the worker
    is terminated (docs/DEV_ECM_GUI.md section 5.6).

.DESCRIPTION
    Requirement (user, 2026-09-28): closing the GUI while workers run must NOT silently
    kill them -- it must ask, and when the user agrees the worker has to write a
    checkpoint first, so at most the work after the last checkpoint is lost.

    Covered here, all against the real window:

      [A] exit_confirm = ask: WM_CLOSE does NOT close the window; it opens the modal
          (trace: "exit: confirmation requested"); Escape cancels it ("exit: cancelled by
          the user") and the GUI keeps running with its worker.
      [B] exit_confirm = ask + a REAL ecm_cuda worker with ckpt_seconds = 3: after
          confirming, the GUI must report the checkpoint and only then terminate:
            worker 1: graceful stop requested (waiting for a checkpoint, max N s)
            worker 1: checkpoint written (.ecm_ckpt_*.dat), safe to stop
            worker 1: stopping (checkpoint written)
          The checkpoint file's mtime on disk must ALSO be newer than before the close
          (independent of the trace), and the GUI must exit 0 with no leftover process.
      [C] exit_confirm = kill: WM_CLOSE terminates immediately (the documented opt-out).

    The modal is driven the way a user would: PostMessage(WM_CLOSE) and then the Enter /
    Escape key messages (the app accepts both, see App::draw_exit_modal).

    Exit code: 0 = all checks passed.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_gui_exit_checkpoint.ps1
#>
param(
    [string]$Exe = "",
    [string]$EcmCuda = "",
    [string]$Sandbox = "",
    [int]$CheckpointSeconds = 3
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
    Write-Host "FAIL: ecm_cuda.exe not found (pass -EcmCuda <path>; this test needs a real driver)" -ForegroundColor Red
    exit 2
}
if (-not $Sandbox) { $Sandbox = Join-Path $PSScriptRoot '_run\gui_exit_checkpoint' }
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

if (-not ("W32ExitFlow" -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class W32ExitFlow {
    public delegate bool EnumProc(IntPtr h, IntPtr p);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr p);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    public const uint WM_CLOSE = 0x0010, WM_KEYDOWN = 0x0100, WM_KEYUP = 0x0101;
    public static IntPtr Find(uint pid, string cls) {
        IntPtr found = IntPtr.Zero;
        EnumWindows((h,p) => { uint wp; GetWindowThreadProcessId(h, out wp);
            if (wp == pid) { var sb = new StringBuilder(128); GetClassName(h, sb, 128);
                if (sb.ToString() == cls) { found = h; return false; } } return true; }, IntPtr.Zero);
        return found;
    }
    public static void PressKey(IntPtr h, int vk) {
        // A plausible scancode makes the key look real to the ImGui Win32 backend.
        int sc = vk == 0x0D ? 0x1C : (vk == 0x1B ? 0x01 : 0x00);
        PostMessage(h, WM_KEYDOWN, (IntPtr)vk, (IntPtr)(1 | (sc << 16)));
        System.Threading.Thread.Sleep(60);
        PostMessage(h, WM_KEYUP, (IntPtr)vk, (IntPtr)(unchecked((int)0xC0000001) | (sc << 16)));
    }
}
'@
}

function Read-TextShared([string]$path) {
    if (-not (Test-Path $path)) { return "" }
    try {
        $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Open,
                                     [System.IO.FileAccess]::Read,
                                     [System.IO.FileShare]::ReadWrite)
        $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)
        $t = $sr.ReadToEnd(); $sr.Close(); $fs.Close(); return $t
    } catch { return "" }
}

$trace = Join-Path (Split-Path -Parent $Exe) 'ecm_gui_trace.log'
function New-Sandbox([string]$name, [string]$exitConfirm, [int]$gracefulMs, [bool]$realDriver) {
    $dir = Join-Path $Sandbox $name
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $enc = New-Object System.Text.UTF8Encoding($false)
    $lines = @(
        'method = gpu', 'backend = auto', 'device = 0', 'gpucurves = 384', 'verbose = 0',
        ('ckpt_seconds = ' + $CheckpointSeconds),
        'save_sync_dir_1 =', 'save_sync_dir_2 =',
        ('tmp_dir = ' + (Join-Path $dir 'saves')),
        ('finished = ' + (Join-Path $dir 'finished.txt')),
        ('worktodo = ' + (Join-Path $dir 'worktodo.txt')),
        '',
        '[GUI]',
        'NumWorkers = 1',
        ('exe = ' + $(if ($realDriver) { Join-Path $dir 'ecm_cuda.exe' } else { Join-Path (Split-Path -Parent $Exe) 'ecm_gui_fake_worker.exe' })),
        'language = english',
        'refresh_hz = 30',
        ('exit_confirm = ' + $exitConfirm),
        ('graceful_stop_ms = ' + $gracefulMs),
        '',
        '[Worker #1]',
        'name = exitflow',
        ('log_file = ' + (Join-Path $dir 'screen_1.log')),
        'autostart = 1'
    )
    if (-not $realDriver) { $lines += 'extra_args = --scenario hang' }
    [System.IO.File]::WriteAllText((Join-Path $dir 'ecm.ini'), (($lines -join "`r`n") + "`r`n"), $enc)
    if ($realDriver) {
        Copy-Item $EcmCuda (Join-Path $dir 'ecm_cuda.exe')
        # 200000 curves of M991/B1=1e4: the run outlives the test, so the exit path is what
        # stops it (a task that finishes on its own would remove the checkpoint).
        [System.IO.File]::WriteAllText((Join-Path $dir 'worktodo.txt'),
            ('ECMSTAGE2=1,2,991,-1,"m991_1e4.save",0,0,4000000' + "`r`n"), $enc)
        # gmp-10.dll sits next to the production driver; copy it if present.
        $gmpCandidates = @((Join-Path (Split-Path -Parent $EcmCuda) 'gmp-10.dll'))
        foreach ($g in $gmpCandidates) { if (Test-Path $g) { Copy-Item $g $dir -Force } }
    } else {
        [System.IO.File]::WriteAllText((Join-Path $dir 'worktodo.txt'), '', $enc)
    }
    return $dir
}

Write-Host "ecm_gui exit flow (confirmation + checkpoint before exit)"
Write-Host ("exe      : " + $Exe)
Write-Host ("ecm_cuda : " + $EcmCuda)
Write-Host ("sandbox  : " + $Sandbox)

# ------------------------------------------------------------------ [A] ask + cancel ----
Write-Host "[A] exit_confirm = ask: WM_CLOSE asks, Escape cancels"
$dirA = New-Sandbox 'cancel' 'ask' 2000 $false
Remove-Item $trace -ErrorAction SilentlyContinue
$pa = Start-Process -FilePath $Exe -ArgumentList @('-ini', (Join-Path $dirA 'ecm.ini'), '--trace') -PassThru
$hwndA = [IntPtr]::Zero
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline -and $hwndA -eq [IntPtr]::Zero) {
    Start-Sleep -Milliseconds 400
    $pa.Refresh(); if ($pa.HasExited) { break }
    $hwndA = [W32ExitFlow]::Find([uint32]$pa.Id, 'ecm_gui')
}
Check "the window appeared" ($hwndA -ne [IntPtr]::Zero)
# The worker autostarts on the first ticks: wait for it instead of racing it.
$logA = ''
$deadline = (Get-Date).AddSeconds(20)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    $pa.Refresh(); if ($pa.HasExited) { break }
    $logA = Read-TextShared $trace
    if ($logA -match 'worker 1: state Running') { break }
}
Check "the worker is running before we close" ($logA -match 'worker 1: state Running') `
      "no 'state Running' in the trace"
[void][W32ExitFlow]::SetForegroundWindow($hwndA)
[void][W32ExitFlow]::PostMessage($hwndA, [W32ExitFlow]::WM_CLOSE, [IntPtr]::Zero, [IntPtr]::Zero)
Start-Sleep -Seconds 2
$pa.Refresh()
Check "WM_CLOSE does not close the window while a worker runs" (-not $pa.HasExited)
$logA = Read-TextShared $trace
Check "the trace reports the confirmation request" ($logA -match 'exit: confirmation requested \(\d+ worker\(s\) running\)') `
      "no 'exit: confirmation requested' line"
[W32ExitFlow]::PressKey($hwndA, 0x1B)      # Escape
Start-Sleep -Seconds 2
$pa.Refresh()
Check "Escape cancels the modal and keeps the GUI running" (-not $pa.HasExited)
$logA = Read-TextShared $trace
Check "the trace reports the cancellation" ($logA -match 'exit: cancelled by the user') `
      "no 'exit: cancelled by the user' line"
Check "the worker is still supervised after cancel" ($logA -notmatch 'worker 1: stopping') `
      "the worker was stopped even though the user cancelled"
if (-not $pa.HasExited) { $pa.CloseMainWindow() | Out-Null; Start-Sleep -Seconds 2 }
$pa.Refresh(); if (-not $pa.HasExited) { $pa.Kill() }
Start-Sleep -Seconds 2
$stray = @(Get-Process -Name 'ecm_gui_fake_worker' -ErrorAction SilentlyContinue)
Check "no fake worker left behind after the cancelled close" ($stray.Count -eq 0) `
      ("still running: " + (($stray | ForEach-Object { $_.Id }) -join ','))

# ------------------------------------------------- [B] ask + confirm + real checkpoint ---
Write-Host "[B] exit_confirm = ask + real driver: the checkpoint is written before the exit"
$dirB = New-Sandbox 'checkpoint' 'ask' 90000 $true
Remove-Item $trace -ErrorAction SilentlyContinue
$pb = Start-Process -FilePath $Exe -ArgumentList @('-ini', (Join-Path $dirB 'ecm.ini'), '--trace') -PassThru
$hwndB = [IntPtr]::Zero
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline -and $hwndB -eq [IntPtr]::Zero) {
    Start-Sleep -Milliseconds 400
    $pb.Refresh(); if ($pb.HasExited) { break }
    $hwndB = [W32ExitFlow]::Find([uint32]$pb.Id, 'ecm_gui')
}
Check "the window appeared (real driver)" ($hwndB -ne [IntPtr]::Zero)
# Wait for the worker to really run and to have checkpointed at least once.
$deadline = (Get-Date).AddSeconds(60)
$ckptBefore = 0
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 1
    $pb.Refresh(); if ($pb.HasExited) { break }
    $ckpts = @(Get-ChildItem $dirB -Filter '.ecm_ckpt_*.dat' -Force -ErrorAction SilentlyContinue)
    if ($ckpts.Count -gt 0) {
        $ckptBefore = ($ckpts | Measure-Object -Property LastWriteTimeUtc -Maximum).Maximum.Ticks
        break
    }
}
$logB = Read-TextShared $trace
Check "the real driver is running" ($logB -match 'worker 1: state Running')
Check "the driver produced progress lines" ($logB -match 'worker 1: progress pct=') `
      "no progress line in the trace"
Check "a checkpoint file exists before we close" ($ckptBefore -gt 0) `
      ("no .ecm_ckpt_*.dat in " + $dirB)
$workerPid = 0
$m = [regex]::Match($logB, 'worker 1: autostart pid=(\d+)')
if ($m.Success) { $workerPid = [int]$m.Groups[1].Value }
Check "we know the driver's pid" ($workerPid -gt 0)

Copy-Item $trace (Join-Path $dirB 'trace_before_close.log') -Force -ErrorAction SilentlyContinue
[void][W32ExitFlow]::SetForegroundWindow($hwndB)
[void][W32ExitFlow]::PostMessage($hwndB, [W32ExitFlow]::WM_CLOSE, [IntPtr]::Zero, [IntPtr]::Zero)
Start-Sleep -Seconds 2
$pb.Refresh()
Check "the modal is up before we confirm" (-not $pb.HasExited)
[W32ExitFlow]::PressKey($hwndB, 0x0D)      # Enter = "Stop and quit"
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 1
    $pb.Refresh(); if ($pb.HasExited) { break }
}
$pb.Refresh()
Check "the GUI exited after the confirmed stop" $pb.HasExited
if ($pb.HasExited) { Check "it exited with code 0" ($pb.ExitCode -eq 0) ("exit=" + $pb.ExitCode) }
$logB = Read-TextShared $trace
Check "the confirmation was accepted" ($logB -match 'exit: confirmed by the user')
Check "the graceful stop was requested" `
      ($logB -match 'worker 1: graceful stop requested \(waiting for a checkpoint') `
      "no 'graceful stop requested' line"
# THE requirement: a checkpoint is written before the worker is terminated.
Check "the GUI saw the fresh checkpoint" ($logB -match 'worker 1: checkpoint written \([^)]*\.ecm_ckpt_[^)]*\), safe to stop') `
      "no 'checkpoint written (...)' line"
Check "the worker was stopped only after the checkpoint" `
      ($logB -match 'worker 1: stopping \(checkpoint written\)') `
      "the stop line does not say 'checkpoint written'"
$ckptAfter = 0
$ckpts = @(Get-ChildItem $dirB -Filter '.ecm_ckpt_*.dat' -Force -ErrorAction SilentlyContinue)
if ($ckpts.Count -gt 0) {
    $ckptAfter = ($ckpts | Measure-Object -Property LastWriteTimeUtc -Maximum).Maximum.Ticks
}
Check "the checkpoint on disk is newer than before the close" ($ckptAfter -gt $ckptBefore) `
      ("before=" + $ckptBefore + " after=" + $ckptAfter)
# Give the termination a moment: the job object kills the driver as the GUI's handle closes.
$deadline = (Get-Date).AddSeconds(15)
$alive = @()
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 1
    $alive = @()
    if ($workerPid -gt 0) { $alive = @(Get-Process -Id $workerPid -ErrorAction SilentlyContinue) }
    if ($alive.Count -eq 0) { break }
}
Check "the driver process is gone" ($alive.Count -eq 0) ("pid " + $workerPid + " still alive")

# ------------------------------------------------------------------ [C] kill policy -----
Write-Host "[C] exit_confirm = kill: the documented opt-out still terminates immediately"
$dirC = New-Sandbox 'kill' 'kill' 2000 $false
Remove-Item $trace -ErrorAction SilentlyContinue
$pc = Start-Process -FilePath $Exe -ArgumentList @('-ini', (Join-Path $dirC 'ecm.ini'), '--trace') -PassThru
$hwndC = [IntPtr]::Zero
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline -and $hwndC -eq [IntPtr]::Zero) {
    Start-Sleep -Milliseconds 400
    $pc.Refresh(); if ($pc.HasExited) { break }
    $hwndC = [W32ExitFlow]::Find([uint32]$pc.Id, 'ecm_gui')
}
# Wait for the worker, otherwise the close finds "nothing running" and skips the policy.
$deadline = (Get-Date).AddSeconds(20)
$logC = ''
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    $pc.Refresh(); if ($pc.HasExited) { break }
    $logC = Read-TextShared $trace
    if ($logC -match 'worker 1: state Running') { break }
}
Check "the kill-policy worker is running" ($logC -match 'worker 1: state Running')
[void][W32ExitFlow]::PostMessage($hwndC, [W32ExitFlow]::WM_CLOSE, [IntPtr]::Zero, [IntPtr]::Zero)
if (-not $pc.WaitForExit(20000)) { $pc.Kill(); Check "kill policy exits promptly" $false "had to kill it" }
else { Check "kill policy exits promptly" ($pc.ExitCode -eq 0) ("exit=" + $pc.ExitCode) }
$logC = Read-TextShared $trace
Check "the kill policy is traced" ($logC -match 'exit_confirm=kill')
Check "no confirmation modal with exit_confirm = kill" ($logC -notmatch 'exit: confirmation requested')
Start-Sleep -Seconds 2
$stray = @(Get-Process -Name 'ecm_gui_fake_worker' -ErrorAction SilentlyContinue)
Check "no worker left behind with the kill policy" ($stray.Count -eq 0)

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
