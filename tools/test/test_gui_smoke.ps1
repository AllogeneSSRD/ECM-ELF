#Requires -Version 5.1
<#
.SYNOPSIS
    Smoke test for ecm_gui milestone M1 (docs/usage/GUI.md).

.DESCRIPTION
    Runs the real GUI once, against a sandbox ecm.ini, and checks the two things a
    headless self-test cannot:

      1. the window really appears (Win32 + Direct3D 11 + ImGui init succeed), and
      2. closing it persists the layout back into ecm.ini [GUI] (window= + the
         dock_layout blob), while the driver keys the GUI does not own are kept
         byte-for-byte.

    The window appears on the desktop for a few seconds, then the script closes it
    gracefully (CloseMainWindow -> WM_CLOSE) so the shutdown path runs.

    Exit code: 0 = all checks passed.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_gui_smoke.ps1
#>
param(
    [string]$Exe = "",
    [string]$Sandbox = "",
    [int]$ShowSeconds = 5
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
    Write-Host "FAIL: ecm_gui.exe not found (pass -Exe <path>; build it with 'cmake --build <dir> --target ecm_gui')" -ForegroundColor Red
    exit 2
}
if (-not $Sandbox) { $Sandbox = Join-Path $PSScriptRoot '_run\gui_smoke' }
if (Test-Path $Sandbox) { Remove-Item $Sandbox -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Sandbox | Out-Null

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
$script:tracePath = Join-Path (Split-Path -Parent $Exe) 'ecm_gui_trace.log'

# ---- Win32 helpers (declared once, used by every step below) --------------------
if (-not ("Win32GuiSmoke" -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class Win32GuiSmoke {
    public delegate bool EnumProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr p);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint msg, IntPtr wp, IntPtr lp);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern uint GetDpiForWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
    public const uint WM_CLOSE = 0x0010;
    public const uint WM_SYSCOMMAND = 0x0112;
    public const int SC_MINIMIZE = 0xF020;
    public const int SC_RESTORE = 0xF120;
    public const int SW_RESTORE = 9;
    public static IntPtr FindByClass(uint pid, string cls) {
        IntPtr found = IntPtr.Zero;
        EnumWindows((h, p) => {
            uint winPid; GetWindowThreadProcessId(h, out winPid);
            if (winPid != pid) return true;
            var sb = new StringBuilder(256); GetClassName(h, sb, sb.Capacity);
            if (sb.ToString() == cls) { found = h; return false; }
            return true;
        }, IntPtr.Zero);
        return found;
    }
    public static string Title(IntPtr h) {
        var sb = new StringBuilder(512); GetWindowText(h, sb, sb.Capacity); return sb.ToString();
    }
    public static void Close(IntPtr h) { PostMessage(h, WM_CLOSE, IntPtr.Zero, IntPtr.Zero); }
}
'@
}
function Check([string]$name, [bool]$ok, [string]$detail = "") {
    if ($ok) {
        $script:pass++
        Write-Host ("  [ok]   " + $name)
    } else {
        $script:fail++
        Write-Host ("  [FAIL] " + $name + $(if ($detail) { " -- " + $detail } else { "" })) -ForegroundColor Red
    }
}

$iniPath = Join-Path $Sandbox 'ecm.ini'
# The UI language under test: it drives the ini, the localization trace line and the
# font assertions in step [7] (a CJK language must bring a CJK-capable system font).
$Language = 'chineseSimplified'
$iniLines = @(
    '# sandbox ini: driver keys the GUI must not touch',
    'method = gpu',
    'device = 0',
    'gpucurves = 384',
    ('tmp_dir = ' + (Join-Path $Sandbox 'saves')),
    '',
    '[Worker #1]',
    'name = first',
    '',
    '[Worker #2]',
    'device = 0',
    'name = second',
    '[GUI]',
    'NumWorkers = 2',
    ('language = ' + $Language),
    'refresh_hz = 15'
)
$enc = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($iniPath, (($iniLines -join "`r`n") + "`r`n"), $enc)

# Copy the localization next to the executable (the GUI looks there first). The GUI
# sources live in src/gui/ -- not at the repo root (it moved there in M8).
$locSrc = Join-Path $repoRoot 'src\gui\localization'
$locDst = Join-Path (Split-Path -Parent $Exe) 'localization'
if ((Test-Path $locSrc) -and -not (Test-Path (Join-Path $locDst 'english.xml'))) {
    Copy-Item $locSrc $locDst -Recurse -Force
}

Write-Host ("exe     : " + $Exe)
Write-Host ("sandbox : " + $Sandbox)

# ---------------------------------------------------------------- 1. window ----
Write-Host "[1] the GUI starts and shows a window"
$proc = Start-Process -FilePath $Exe -ArgumentList @('-ini', $iniPath, '--trace') -PassThru
$deadline = (Get-Date).AddSeconds(30)
$title = ''
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    $proc.Refresh()
    if ($proc.HasExited) { break }
    if ($proc.MainWindowHandle -ne 0) { $title = $proc.MainWindowTitle; break }
}
Check "process is alive" (-not $proc.HasExited) ("exit code " + $proc.ExitCode)
Check "a main window exists" ($proc.MainWindowHandle -ne 0)
Check "window reports a title" (-not [string]::IsNullOrEmpty($title)) $title

# ------------------------------------------------------- 2. graceful close ----
Write-Host ("[2] close it gracefully after {0}s (shutdown path must save the layout)" -f $ShowSeconds)

# ------------------------------------------- 1c. messages while minimized ----
# Regression guard for the reported bug: with the old loop order (iconic check BEFORE the
# message pump) nothing posted to the window was dispatched while minimized, so the GUI
# looked dead. A posted WM_CLOSE must therefore terminate the app *in the minimized
# state* -- that only works if the pump keeps running.
Write-Host "[1c] a minimized GUI still processes window messages"
$proc1c = Start-Process -FilePath $Exe -ArgumentList @('-ini', $iniPath, '--trace') -PassThru
$hwnd1c = [IntPtr]::Zero
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline -and $hwnd1c -eq [IntPtr]::Zero) {
    Start-Sleep -Milliseconds 400
    $proc1c.Refresh()
    if ($proc1c.HasExited) { break }
    $hwnd1c = [Win32GuiSmoke]::FindByClass([uint32]$proc1c.Id, 'ecm_gui')
}
Check "a second instance started" ($hwnd1c -ne [IntPtr]::Zero)
if ($hwnd1c -ne [IntPtr]::Zero) {
    [void][Win32GuiSmoke]::PostMessage($hwnd1c, [Win32GuiSmoke]::WM_SYSCOMMAND,
                                       [IntPtr][Win32GuiSmoke]::SC_MINIMIZE, [IntPtr]::Zero)
    $iconic1c = $false
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 200
        if ([Win32GuiSmoke]::IsIconic($hwnd1c)) { $iconic1c = $true; break }
    }
    Check "the second instance is minimized" $iconic1c
    [Win32GuiSmoke]::Close($hwnd1c)
    $exited1c = $proc1c.WaitForExit(20000)
    if (-not $exited1c) { $proc1c.Kill() }
    Check "a minimized GUI still reacts to WM_CLOSE (message pump alive)" $exited1c
    if ($exited1c) {
        Check "and it exits cleanly" ($proc1c.ExitCode -eq 0) ("exit code " + $proc1c.ExitCode)
    }
}
Start-Sleep -Seconds $ShowSeconds

# Close the window the app itself registered (class "ecm_gui"). Process.MainWindowHandle
# is not reliable here: with multi-viewport there are several top-level windows and it
# may pick an ImGui helper window, whose WM_CLOSE does not stop the application.
# (The Win32 helper type itself is declared near the top of this script, so every step
# can use it.)
$hwnd = [Win32GuiSmoke]::FindByClass([uint32]$proc.Id, 'ecm_gui')
Check "found the app's own window class (ecm_gui)" ($hwnd -ne [IntPtr]::Zero)
$rect1 = $null
if ($hwnd -ne [IntPtr]::Zero) {
    Write-Host ("       window title: '" + [Win32GuiSmoke]::Title($hwnd) + "'")
    $r1 = New-Object Win32GuiSmoke+RECT
    [void][Win32GuiSmoke]::GetWindowRect($hwnd, [ref]$r1)
    $rect1 = @($r1.Left, $r1.Top, ($r1.Right - $r1.Left), ($r1.Bottom - $r1.Top))
    # The GUI is per-monitor DPI aware, PowerShell is not: the same window is reported
    # in physical pixels by the GUI and in virtualized pixels here. Scale before
    # comparing (100% displays have dpi 96 and this is a no-op).
    $dpi = [Win32GuiSmoke]::GetDpiForWindow($hwnd)
    if ($dpi -eq 0) { $dpi = 96 }
    $script:dpiScale = [double]$dpi / 96.0
    Write-Host ("       live rectangle: {0}  (dpi {1}, scale {2:N2})" -f ($rect1 -join ','), $dpi, $script:dpiScale)

    # ---- minimize / restore -------------------------------------------------
    # The bug the user hit ("minimized, cannot be brought back"): the frame loop skipped
    # the message pump while the window was iconic, so nothing sent to the window was
    # ever dispatched. Two things are checked:
    #   (a) the window can be brought back (ShowWindow(SW_RESTORE) -- the same
    #       kernel-side path the taskbar uses; a *posted* SC_RESTORE from another process
    #       is ignored by Windows, so that is not a valid probe);
    #   (b) the message pump really runs while minimized: the process must react to a
    #       posted WM_CLOSE in that state (step 1c below), which the old order could not.
    Write-Host "[1b] minimize, then bring the window back"
    [void][Win32GuiSmoke]::PostMessage($hwnd, [Win32GuiSmoke]::WM_SYSCOMMAND,
                                       [IntPtr][Win32GuiSmoke]::SC_MINIMIZE, [IntPtr]::Zero)
    $minimized = $false
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 200
        if ([Win32GuiSmoke]::IsIconic($hwnd)) { $minimized = $true; break }
    }
    Check "the window minimizes" $minimized
    [void][Win32GuiSmoke]::ShowWindow($hwnd, [Win32GuiSmoke]::SW_RESTORE)
    $restored = $false
    for ($i = 0; $i -lt 25; $i++) {
        Start-Sleep -Milliseconds 200
        if (-not [Win32GuiSmoke]::IsIconic($hwnd)) { $restored = $true; break }
    }
    Check "the window can be brought back" $restored
    Check "the restored window is visible" ([Win32GuiSmoke]::IsWindowVisible($hwnd))
    $proc.Refresh()
    Check "the process survived minimize/restore" (-not $proc.HasExited)

    [Win32GuiSmoke]::Close($hwnd)
} else {
    [void]$proc.CloseMainWindow()
}
if (-not $proc.WaitForExit(25000)) {
    $proc.Kill()
    Check "process exited after WM_CLOSE" $false "had to kill it"
} else {
    Check "process exited after WM_CLOSE" ($proc.ExitCode -eq 0) ("exit code " + $proc.ExitCode)
}
# Keep run 1's trace: later steps start more instances and the GUI truncates the file
# on every start, and a short run may not reach the frame that traces the panel rects.
$script:traceRun1 = Join-Path $Sandbox 'trace_run1.log'
Copy-Item $script:tracePath $script:traceRun1 -Force -ErrorAction SilentlyContinue

# ------------------------------------------------- 3. layout in ecm.ini ----
Write-Host "[3] [GUI] keys written back, driver keys preserved"
$after = Get-Content -LiteralPath $iniPath -Raw
Check "window= rectangle persisted"   ($after -match '(?m)^window\s*=\s*-?\d+,-?\d+,\d+,\d+')
Check "dock_layout blob persisted"    ($after -match '(?m)^dock_layout\s*=\s*\S')
Check "NumWorkers preserved (2)"      ($after -match '(?m)^NumWorkers\s*=\s*2\s*$')
Check "refresh_hz preserved (15)"     ($after -match '(?m)^refresh_hz\s*=\s*15\s*$')
Check "language kept"                 ($after -match '(?m)^language\s*=\s*chineseSimplified\s*$')
Check "driver key gpucurves untouched" ($after -match '(?m)^gpucurves\s*=\s*384\s*$')
Check "driver key tmp_dir untouched"  ($after -match ('(?m)^tmp_dir\s*=\s*' + [regex]::Escape((Join-Path $Sandbox 'saves'))))
Check "worker name kept"              ($after -match '(?m)^name\s*=\s*first\s*$')
Check "sandbox comment kept"          ($after -match '# sandbox ini: driver keys the GUI must not touch')
Check "backup generation written"     (Test-Path ($iniPath + '.bak'))

# The saved ini must still be readable by the driver: run its own queue manager
# against it (a bogus task line keeps the GPU out of it).
$ecmCuda = Join-Path $repoRoot 'build_cuda_cmake\ecm_cuda.exe'
if (Test-Path $ecmCuda) {
    Write-Host "[4] the driver can still read the rewritten ini"
    $todo = Join-Path $Sandbox 'worktodo.txt'
    [System.IO.File]::WriteAllText($todo, "ECMSTAGE2=garbage`r`n", $enc)
    $iniText = [System.IO.File]::ReadAllText($iniPath)
    if ($iniText -notmatch '(?m)^worktodo\s*=') {
        [System.IO.File]::AppendAllText($iniPath, ("worktodo = " + $todo + "`r`n"), $enc)
    }
    $out = (& cmd /c ('"{0}" -ini "{1}" --worker 1 < NUL 2>&1' -f $ecmCuda, $iniPath) | Out-String)
    Check "driver starts with the GUI-written ini" ($out -match 'ECM queue manager')
    Check "driver read NumWorkers/worker 1"        ($out -match 'worker : 1')
} else {
    Write-Host "[4] skipped: ecm_cuda.exe not built here"
}

# ------------------------------------------------- 5. geometry restored ----
Write-Host "[5] second run restores the saved window geometry"
$m = [regex]::Match($after, '(?m)^window\s*=\s*(-?\d+),(-?\d+),(\d+),(\d+)\s*$')
Check "window= parses as x,y,w,h" $m.Success
if ($m.Success -and $rect1) {
    $saved = @([int]$m.Groups[1].Value, [int]$m.Groups[2].Value,
               [int]$m.Groups[3].Value, [int]$m.Groups[4].Value)
    Write-Host ("       saved by run 1: {0}   run 1 was live at: {1} (x{2:N2} = {3})" -f `
                ($saved -join ','), ($rect1 -join ','), $script:dpiScale,
                (($rect1 | ForEach-Object { [Math]::Round($_ * $script:dpiScale) }) -join ','))
    # Capture check: what the GUI wrote must be its own live rectangle. The GUI's
    # coordinate space is physical pixels, this shell's is virtualized -> scale.
    $cap = $true
    for ($i = 0; $i -lt 4; $i++) {
        $expect = [Math]::Round($rect1[$i] * $script:dpiScale)
        if ([Math]::Abs($saved[$i] - $expect) -gt 3) { $cap = $false }
    }
    Check "saved window= matches run 1's live rectangle" $cap (($saved -join ',') + " vs scaled " + (($rect1 | ForEach-Object { [Math]::Round($_ * $script:dpiScale) }) -join ','))

    $proc2 = Start-Process -FilePath $Exe -ArgumentList @('-ini', $iniPath, '--trace') -PassThru
    $hwnd2 = [IntPtr]::Zero
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline -and $hwnd2 -eq [IntPtr]::Zero) {
        Start-Sleep -Milliseconds 400
        $proc2.Refresh()
        if ($proc2.HasExited) { break }
        $hwnd2 = [Win32GuiSmoke]::FindByClass([uint32]$proc2.Id, 'ecm_gui')
    }
    Check "second window appeared" ($hwnd2 -ne [IntPtr]::Zero)
    if ($hwnd2 -ne [IntPtr]::Zero) {
        $r2 = New-Object Win32GuiSmoke+RECT
        [void][Win32GuiSmoke]::GetWindowRect($hwnd2, [ref]$r2)
        $rect2 = @($r2.Left, $r2.Top, ($r2.Right - $r2.Left), ($r2.Bottom - $r2.Top))
        Write-Host ("       run 2 live rectangle: " + ($rect2 -join ','))
        $ok = $true
        for ($i = 0; $i -lt 4; $i++) { if ([Math]::Abs($rect1[$i] - $rect2[$i]) -gt 2) { $ok = $false } }
        Check "second run's rectangle equals the first run's (restored)" $ok (($rect1 -join ',') + " vs " + ($rect2 -join ','))
        [Win32GuiSmoke]::Close($hwnd2)
        if (-not $proc2.WaitForExit(25000)) { $proc2.Kill(); Check "second run exits cleanly" $false "had to kill" }
        else { Check "second run exits cleanly" ($proc2.ExitCode -eq 0) ("exit code " + $proc2.ExitCode) }
    } else {
        $proc2.Kill()
    }
}

# ------------------------------------------------- 6. initial layout sanity ----
# The panels must be arranged inside the client area and must not overlap (equal
# rectangles are intentional: those panels share one dock node as tabs). This is the
# measurable version of "the windows are not a pile in the middle".
Write-Host "[6] the panels are arranged inside the window and do not overlap"
$log = Read-TextShared $script:traceRun1
$vp = [regex]::Match($log, 'layout: viewport pos=\((-?\d+),(-?\d+)\) size=(\d+)x(\d+)')
Check "the trace reports the viewport rect" $vp.Success
if ($vp.Success) {
    $vx = [int]$vp.Groups[1].Value; $vy = [int]$vp.Groups[2].Value
    $vw = [int]$vp.Groups[3].Value; $vh = [int]$vp.Groups[4].Value
    $panels = @()
    foreach ($m in [regex]::Matches($log, 'layout: (\w+) (###\S+) x=(-?\d+) y=(-?\d+) w=(\d+) h=(\d+)')) {
        if ($m.Groups[2].Value -eq '###ecm_gui_host') { continue }
        $panels += [pscustomobject]@{
            id = $m.Groups[2].Value
            x  = [int]$m.Groups[3].Value; y = [int]$m.Groups[4].Value
            w  = [int]$m.Groups[5].Value; h = [int]$m.Groups[6].Value
        }
    }
    Check "all five panels were laid out (4 + 2 log tabs)" ($panels.Count -eq 6) ("found=" + $panels.Count)
    $outside = @($panels | Where-Object {
        $_.x -lt $vx -or $_.y -lt $vy -or ($_.x + $_.w) -gt ($vx + $vw + 2) -or ($_.y + $_.h) -gt ($vy + $vh + 2)
    })
    Check "every panel is inside the client area" ($outside.Count -eq 0) `
          (($outside | ForEach-Object { $_.id + "@" + $_.x + "," + $_.y + "+" + $_.w + "x" + $_.h }) -join ' ')
    $sized = @($panels | Where-Object { $_.w -lt 50 -or $_.h -lt 50 })
    Check "no panel is collapsed to nothing" ($sized.Count -eq 0) (($sized | ForEach-Object { $_.id }) -join ' ')
    # Pairwise overlap, ignoring panels that share a rectangle exactly (tabs).
    $overlaps = @()
    for ($i = 0; $i -lt $panels.Count; $i++) {
        for ($j = $i + 1; $j -lt $panels.Count; $j++) {
            $a = $panels[$i]; $b = $panels[$j]
            if ($a.x -eq $b.x -and $a.y -eq $b.y -and $a.w -eq $b.w -and $a.h -eq $b.h) { continue }
            $ix = [Math]::Min($a.x + $a.w, $b.x + $b.w) - [Math]::Max($a.x, $b.x)
            $iy = [Math]::Min($a.y + $a.h, $b.y + $b.h) - [Math]::Max($a.y, $b.y)
            if ($ix -gt 2 -and $iy -gt 2) { $overlaps += ($a.id + " vs " + $b.id) }
        }
    }
    Check "no two panels overlap" ($overlaps.Count -eq 0) ($overlaps -join '; ')
    Check "the ini records the layout version" ($after -match '(?m)^dock_layout_ver\s*=\s*\d+')
    # The agreed arrangement: left column = workers (+detail tab) over the output tabs,
    # right column = GPU over results.
    $want = @{ '###workers' = 'left'; '###detail' = 'left'; '###log1' = 'left'; '###gpu' = 'right'; '###results' = 'right' }
    $cols = @{}
    foreach ($p in $panels) { if ($want.ContainsKey($p.id)) { $cols[$p.id] = $p.x } }
    $leftXs = @($cols.Keys | Where-Object { $want[$_] -eq 'left' } | ForEach-Object { $cols[$_] } | Sort-Object -Unique)
    $rightXs = @($cols.Keys | Where-Object { $want[$_] -eq 'right' } | ForEach-Object { $cols[$_] } | Sort-Object -Unique)
    Check "the left panels share one column" ($leftXs.Count -eq 1) ($leftXs -join ',')
    Check "the right panels share one column" ($rightXs.Count -eq 1) ($rightXs -join ',')
    if ($leftXs.Count -eq 1 -and $rightXs.Count -eq 1) {
        Check "the left column comes first and is the wider one" ($leftXs[0] -lt $rightXs[0])
        $leftW = ($panels | Where-Object { $_.id -eq '###workers' }).w
        $rightW = ($panels | Where-Object { $_.id -eq '###gpu' }).w
        Check "the left column is ~60 % / the right ~40 %" `
              ([Math]::Abs($leftW / ($leftW + $rightW) - 0.60) -le 0.05) `
              ("left=" + $leftW + " right=" + $rightW)
    }
    $wTop = ($panels | Where-Object { $_.id -eq '###workers' }).y
    $wBot = ($panels | Where-Object { $_.id -eq '###log1' }).y
    Check "workers are above the output tabs" ($wTop -lt $wBot) ($wTop.ToString() + " vs " + $wBot)
    $gTop = ($panels | Where-Object { $_.id -eq '###gpu' }).y
    $gBot = ($panels | Where-Object { $_.id -eq '###results' }).y
    Check "GPU is above results" ($gTop -lt $gBot) ($gTop.ToString() + " vs " + $gBot)

    # The Workers table must fit its panel: the progress bar and the ETA column used to be
    # pushed out of view by the (very long) Task column (user report 2026-09-28).
    $tbls = [regex]::Matches($log, "table: workers right=([\d.]+) left=([\d.]+) progress_x=([\d.]+) progress_w=([\d.]+) .* fits=(\d)")
    Check "the trace reports the Workers table geometry" ($tbls.Count -gt 0)
    if ($tbls.Count -gt 0) {
        $t = $tbls[$tbls.Count - 1]
        Write-Host ("       table right=" + $t.Groups[1].Value + " progress=" +
                    $t.Groups[3].Value + "+" + $t.Groups[4].Value + " fits=" + $t.Groups[5].Value)
        Check "the table fits its panel at this window size" ($t.Groups[5].Value -eq '1')
        # 885 px panel in this test; the user's layout gives the Workers panel 1240 px
        # (~475 px bar). 90 px is the floor below which the bar stops being readable
        # (a regression once left it at 45 px).
        Check "the progress bar is wide enough to read" ([double]$t.Groups[4].Value -ge 90) `
              ("progress_w=" + $t.Groups[4].Value)
    }
}

Write-Host "[7] the font is DPI scaled and can draw the UI language"
$fonthdr = [regex]::Match($log, "font: (\S+) at ([\d.]+) px \(dpi x([\d.]+), ([^,)]*)")
Check "the trace reports the chosen font and size" $fonthdr.Success
if ($fonthdr.Success) {
    $size = [double]$fonthdr.Groups[2].Value
    $scale = [double]$fonthdr.Groups[3].Value
    Write-Host ("       font=" + $fonthdr.Groups[1].Value + " size=" + $size + "px dpi=x" + $scale +
                " (" + $fonthdr.Groups[4].Value + ")")
    # Auto size is 15 px * DPI scale and is deliberately NOT rounded (22.5 at 150 %),
    # because rounding it is what makes the text look different from the intent.
    Check "the font size follows the DPI scale" ([Math]::Abs($size - (15 * $scale)) -le 1.0) `
          ("size=" + $size + " scale=" + $scale)
    Check "a system outline font is used (not the 13 px built-in)" ($fonthdr.Groups[1].Value -ne '<built-in>')
}
$cjk = [regex]::Match($log, "font: measured latin 'WW'=(\d+)x(\d+) cjk 2-glyphs=(\d+)x(\d+) cjk_ok=(\d) map=(\d)(\d) baked=(\d)(\d) negctl=(\d)(\d)")
Check "the trace reports the measured text metrics" $cjk.Success
if ($cjk.Success) {
    $cw = [int]$cjk.Groups[3].Value
    $ch = [int]$cjk.Groups[4].Value
    Write-Host ("       cjk glyphs " + $cw + "x" + $ch + " cjk_ok=" + $cjk.Groups[5].Value +
                " map=" + $cjk.Groups[6].Value + $cjk.Groups[7].Value +
                " baked=" + $cjk.Groups[8].Value + $cjk.Groups[9].Value +
                " negctl=" + $cjk.Groups[10].Value + $cjk.Groups[11].Value)
    Check "Chinese text has real glyph widths (not tofu)" `
          (($cjk.Groups[5].Value -eq '1') -and $cw -ge 20 -and $ch -ge 10) ("cjk=" + $cw + "x" + $ch)
    # The decisive check: the loaded TTF maps U+6587/U+4EF6 ("文件") AND both were
    # really baked into the atlas at the current size. ImGui would otherwise quietly
    # draw the U+FFFD fallback box -- which also has a nonzero advance.
    Check "the font maps both Chinese codepoints" `
          (($cjk.Groups[6].Value -eq '1') -and ($cjk.Groups[7].Value -eq '1')) `
          ("map=" + $cjk.Groups[6].Value + $cjk.Groups[7].Value)
    Check "both glyphs are baked into the atlas (no fallback box)" `
          (($cjk.Groups[8].Value -eq '1') -and ($cjk.Groups[9].Value -eq '1')) `
          ("baked=" + $cjk.Groups[8].Value + $cjk.Groups[9].Value)
    # Negative control: U+E123 is a private-use codepoint no UI font maps, so it MUST
    # report missing -- otherwise the two checks above would prove nothing.
    Check "the negative control codepoint is reported missing" `
          (($cjk.Groups[10].Value -eq '0') -and ($cjk.Groups[11].Value -eq '0')) `
          ("negctl=" + $cjk.Groups[10].Value + $cjk.Groups[11].Value)
}
$locline = [regex]::Match($log, "localization: dir=(\S+) language=(\S+) keys=(\d+) baseline=(\d+) missing=(\d+) cjk=(\d)")
Check "the trace reports which localization file is in effect" $locline.Success
if ($locline.Success) {
    Write-Host ("       localization=" + $locline.Groups[2].Value + " keys=" + $locline.Groups[3].Value +
                " baseline=" + $locline.Groups[4].Value + " missing=" + $locline.Groups[5].Value)
    Check "the sandbox language file was loaded" ($locline.Groups[2].Value -eq $Language)
    Check "no localization key is missing" ($locline.Groups[5].Value -eq '0')
    Check "the loaded language needs the CJK font" ($locline.Groups[6].Value -eq '1')
}

Write-Host "[8] the first frame completed every draw stage (catches first-frame crashes)"
# A crash inside draw() used to be invisible to the suite: the GUI just died (0xC0000005)
# and the script only saw "process exited early". The GUI traces its draw stages for the
# first frames, so the whole sequence plus "frame 1 rendered" is a hard requirement.
foreach ($stage in @('draw: menu bar ok', 'draw: workers table ok', 'draw: gpu panel ok',
                     'draw: results panel ok', 'draw: detail panel ok',
                     'draw: worker panes ok', 'draw: exit modal ok')) {
    Check ("stage completed: " + $stage) ($log -match [regex]::Escape($stage))
}
Check "frame 1 was rendered" ($log -match 'frame 1 rendered')

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
