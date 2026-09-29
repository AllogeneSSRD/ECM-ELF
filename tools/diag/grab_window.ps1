#Requires -Version 5.1
<#
.SYNOPSIS
    Grab a screenshot of one window (by process + window class) into a PNG.

.DESCRIPTION
    Used to look at the GUI without a human at the screen: find the top-level window of
    a process by class name (e.g. 'ecm_gui'), capture it with PrintWindow into a bitmap
    and save it as PNG. The window is captured as-is (it must be visible; a minimized
    window captures as an empty frame).

.PARAMETER ProcessName
    Process name without extension, e.g. ecm_gui.
.PARAMETER Class
    Window class to capture (default ecm_gui).
.PARAMETER Out
    Output PNG path (default: <repo>\tools\test\_run\<name>.png).

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\diag\grab_window.ps1 -ProcessName ecm_gui -Out shot.png
#>
param(
    [string]$ProcessName = 'ecm_gui',
    [string]$Class = 'ecm_gui',
    [string]$Out = "",
    [int]$WaitSeconds = 10
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

if (-not ("WinGrab" -as [type])) {
    Add-Type @'
using System; using System.Runtime.InteropServices; using System.Text;
public static class WinGrab {
  public delegate bool EnumProc(IntPtr h, IntPtr p);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr p);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern IntPtr GetDC(IntPtr h);
  [DllImport("user32.dll")] public static extern int ReleaseDC(IntPtr h, IntPtr dc);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
  public static IntPtr Find(uint pid, string cls) {
    IntPtr found = IntPtr.Zero;
    EnumWindows((h,p) => { uint wp; GetWindowThreadProcessId(h, out wp);
      if (wp != pid) return true;
      var sb = new StringBuilder(128); GetClassName(h, sb, sb.Capacity);
      if (sb.ToString() == cls) { found = h; return false; } return true; }, IntPtr.Zero);
    return found; }
}
'@
}

$deadline = (Get-Date).AddSeconds($WaitSeconds)
$proc = $null
while ((Get-Date) -lt $deadline) {
    $proc = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
    if ($proc) { break }
    Start-Sleep -Milliseconds 300
}
if (-not $proc) { throw "no $ProcessName process with a window found" }

$hwnd = [WinGrab]::Find([uint32]$proc.Id, $Class)
if ($hwnd -eq [IntPtr]::Zero) { throw "no window of class '$Class' in $ProcessName (pid $($proc.Id))" }

$rect = New-Object WinGrab+RECT
[void][WinGrab]::GetClientRect($hwnd, [ref]$rect)
$w = $rect.Right - $rect.Left
$h = $rect.Bottom - $rect.Top
if ($w -le 0 -or $h -le 0) { throw "window has an empty client area ($w x $h)" }

$bmp = New-Object System.Drawing.Bitmap($w, $h)
$gfx = [System.Drawing.Graphics]::FromImage($bmp)
$hdc = $gfx.GetHdc()
# flag 2 = PW_RENDERFULLCONTENT: needed for DirectComposition/D3D content, otherwise the
# captured frame is blank for a D3D11 window.
$ok = [WinGrab]::PrintWindow($hwnd, $hdc, 2)
$gfx.ReleaseHdc($hdc)
$gfx.Dispose()

if (-not $Out) {
    $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $dir = Join-Path $repoRoot 'tools\test\_run'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $Out = Join-Path $dir ("window_" + $Class + ".png")
}
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()
if (-not $ok) { Write-Host "note: PrintWindow returned false (content may be partial)" }
Write-Host ("saved {0}x{1} -> {2}" -f $w, $h, $Out)
