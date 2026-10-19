#Requires -Version 5.1
param([string]$Exe = "", [int]$Seconds = 6)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Exe) { $Exe = Join-Path $repo 'build_gui\ecm_gui.exe' }
if (-not (Test-Path -LiteralPath $Exe)) { throw "GUI executable not found: $Exe" }
$sandbox = Join-Path $repo ('data\experiments\gui_refresh_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $sandbox -Force | Out-Null
$isolatedExe = Join-Path $sandbox 'ecm_gui.exe'
Copy-Item -LiteralPath $Exe -Destination $isolatedExe
$trace = Join-Path $sandbox 'ecm_gui_trace.log'
$pending = Join-Path $sandbox 'p95_add_pending.txt'

Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class GuiRefreshWin32 {
    public delegate bool EnumProc(IntPtr hwnd, IntPtr arg);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc callback, IntPtr arg);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint pid);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetClassName(IntPtr hwnd, StringBuilder name, int size);
    [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr hwnd, out Rect rect);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr hwnd, uint message, IntPtr w, IntPtr l);
    public static IntPtr Find(uint targetPid) {
        IntPtr found = IntPtr.Zero;
        long largest = 0;
        EnumWindows((hwnd, arg) => {
            uint pid; GetWindowThreadProcessId(hwnd, out pid);
            if (pid != targetPid) return true;
            var name = new StringBuilder(128);
            GetClassName(hwnd, name, name.Capacity);
            if (name.ToString() != "ecm_gui") return true;
            Rect rect;
            if (!GetClientRect(hwnd, out rect)) return true;
            long area = (long)(rect.Right - rect.Left) * (rect.Bottom - rect.Top);
            if (area > largest) { largest = area; found = hwnd; }
            return true;
        }, IntPtr.Zero);
        return found;
    }
}
'@

function Wait-Notice([string]$pattern, [int]$timeoutMs) {
    $deadline = [DateTime]::UtcNow.AddMilliseconds($timeoutMs)
    do {
        if (Test-Path -LiteralPath $trace) {
            $last = Get-Content -LiteralPath $trace | Where-Object { $_ -match 'p95 notice: level=' } | Select-Object -Last 1
            if ($last -match $pattern) { return $last }
        }
        Start-Sleep -Milliseconds 50
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Notice did not become $pattern; trace: $trace"
}

function Run-Gui([int]$hz, [bool]$exercisePending) {
    $ini = Join-Path $sandbox "hz$hz.ini"
    $localization = Join-Path $repo 'localization'
    [System.IO.File]::WriteAllText($ini, @"
p95_worktodo_path = prime95\worktodo.txt
[GUI]
NumWorkers = 1
refresh_hz = $hz
gpu_poll_ms = 1000
localization_dir = $localization
[Worker #1]
autostart = 0
"@, [Text.UTF8Encoding]::new($false))
    $oldNvml = $env:ECM_GUI_NVML
    $env:ECM_GUI_NVML = 'Z:\missing_nvml_for_gui_refresh_test.dll'
    try {
        $process = Start-Process -FilePath $isolatedExe -ArgumentList @('-ini', $ini, '--trace') -WindowStyle Hidden -PassThru
    } finally { $env:ECM_GUI_NVML = $oldNvml }
    try {
        $started = [Diagnostics.Stopwatch]::StartNew()
        Wait-Notice 'level=green parked=0' 10000 | Out-Null
        if ($exercisePending) {
            [IO.File]::WriteAllText($pending, "one`r`ntwo`r`n", [Text.UTF8Encoding]::new($false))
            Wait-Notice 'level=red parked=2' 2500 | Out-Null
            Remove-Item -LiteralPath $pending
            Wait-Notice 'level=green parked=0' 2500 | Out-Null
        }
        while ($started.Elapsed.TotalSeconds -lt $Seconds) { Start-Sleep -Milliseconds 100 }
        $hwnd = [GuiRefreshWin32]::Find([uint32]$process.Id)
        if ($hwnd -eq [IntPtr]::Zero) { throw "GUI window not found for pid $($process.Id)" }
        [GuiRefreshWin32]::PostMessage($hwnd, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
        if (-not $process.WaitForExit(5000)) { throw "GUI did not exit on WM_CLOSE at $hz Hz" }
        if ($process.ExitCode -ne 0) { throw "GUI exit code $($process.ExitCode) at $hz Hz" }
        $match = [regex]::Match([IO.File]::ReadAllText($trace), 'frame loop left after (\d+) frame\(s\)')
        if (-not $match.Success) { throw "Frame count missing from trace: $trace" }
        $frames = [int]$match.Groups[1].Value
        Write-Host "refresh_hz=$hz frames=$frames elapsed=$([math]::Round($started.Elapsed.TotalSeconds, 2))"
        return $frames
    } finally {
        $process.Refresh()
        if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
    }
}

$frames10 = Run-Gui 10 $true
$frames60 = Run-Gui 60 $false
if ($frames10 -lt 10 -or $frames60 -le ($frames10 * 1.2)) {
    throw "refresh_hz did not change the frame rate enough: 10 Hz=$frames10, 60 Hz=$frames60; trace: $trace"
}
Write-Output "PASS: Prime95 pending transitions and refresh_hz frame pacing; evidence: $sandbox"
