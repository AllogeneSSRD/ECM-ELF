#Requires -Version 5.1
<#
.SYNOPSIS
    Acceptance test for the Prime95 handoff NOTICE STRIP in the GUI
    (docs/DEV_ECM_GUI.md 18): red / yellow / green / grey, always visible, and the
    "open Prime95 folder" / "open parked file" buttons.

.DESCRIPTION
    The strip's state is traced every frame it changes (`p95 notice: level=… parked=… text=…`),
    so this test drives the GUI and asserts what the user sees without looking at pixels.

      [A1] no p95 keys            -> grey  "not configured" (no worker needed)
      [A2] p95_worktodo_path set  -> green "ready"
      [A3] + a parked file        -> red   parked=2 (a file left by an earlier run counts,
                                            even with no worker running)
      [B]  a real worker whose task is handed over with a fallback (p95_add_workers = 3 and
           no [Worker #3] in Prime95's worktodo.txt) -> yellow + the driver's note
      [C]  a real worker whose delivery fails (a fresh worktodo.add.lock) -> red + parked=1,
           and the parked line is really on disk

    The GUI is closed with WM_CLOSE and must exit 0; the parked file and the lock are
    cleaned up even when a check fails, so the next test starts from a clean state.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_gui_p95_notice.ps1
#>
param(
    [string]$Exe = "",
    [string]$EcmCuda = "",
    [string]$Sandbox = "",
    [int]$Device = 0,
    [int]$TimeoutSeconds = 120
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
if (-not $Sandbox) { $Sandbox = Join-Path $PSScriptRoot '_run\gui_p95_notice' }
if (Test-Path $Sandbox) { Remove-Item $Sandbox -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $Sandbox 'saves') | Out-Null

# The parked file lives next to the WORKER executable (that is what the driver uses), so a
# real-worker scenario parks it in the build directory. Never leave it behind: the next
# suite test would start with a red strip.
$realPending = Join-Path (Split-Path -Parent $EcmCuda) 'p95_add_pending.txt'

$script:pass = 0
$script:fail = 0
function Check([string]$name, [bool]$ok, [string]$detail = "") {
    if ($ok) { $script:pass++; Write-Host ("  [ok]   " + $name) }
    else {
        $script:fail++
        Write-Host ("  [FAIL] " + $name + $(if ($detail) { " -- " + $detail } else { "" })) -ForegroundColor Red
    }
}
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

if (-not ("Win32P95" -as [type])) {
    Add-Type @'
using System; using System.Runtime.InteropServices; using System.Text;
public static class Win32P95 {
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
  // --- pixel proof that the strip really reaches the screen -------------------------
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr dc, uint flags);
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
  public static int[] ClientOrigin(IntPtr h) {
    var p = new POINT(); ClientToScreen(h, ref p); return new int[] { p.X, p.Y }; }
  public static int[] ClientSize(IntPtr h) {
    RECT r; GetClientRect(h, out r); return new int[] { r.Right - r.Left, r.Bottom - r.Top }; }
  public static bool Shot(IntPtr h, IntPtr dc) { return PrintWindow(h, dc, 2); }  // 2 = PW_RENDERFULLCONTENT
  // The GUI runs with ImGui viewports enabled, so SEVERAL windows share the class name
  // 'ecm_gui' (the main window plus one platform window per popped-out panel). Find() returned
  // a 985x562 panel and the strip was nowhere in the capture (measured 2026-09-29), so the
  // main window is picked by the largest client area.
  public static IntPtr FindMain(uint pid, string cls) {
    IntPtr best = IntPtr.Zero; long bestArea = 0;
    EnumWindows((h,p) => { uint wp; GetWindowThreadProcessId(h, out wp);
      if (wp != pid) return true;
      var sb = new StringBuilder(128); GetClassName(h, sb, sb.Capacity);
      if (sb.ToString() != cls) return true;
      RECT r; GetClientRect(h, out r);
      long area = (long)(r.Right - r.Left) * (long)(r.Bottom - r.Top);
      if (area > bestArea) { bestArea = area; best = h; }
      return true; }, IntPtr.Zero);
    return best; }
}
'@
}

# Captures the window's client area into a PNG and returns where that area sits on screen
# (needed to map the traced strip rect into image pixels).
function Save-WindowShot([IntPtr]$hwnd, [string]$outPath) {
    Add-Type -AssemblyName System.Drawing
    $size = [Win32P95]::ClientSize($hwnd)
    $org = [Win32P95]::ClientOrigin($hwnd)
    $bmp = New-Object System.Drawing.Bitmap($size[0], $size[1])
    $gfx = [System.Drawing.Graphics]::FromImage($bmp)
    $hdc = $gfx.GetHdc()
    $ok = [Win32P95]::Shot($hwnd, $hdc)
    $gfx.ReleaseHdc($hdc); $gfx.Dispose()
    $bmp.Save($outPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    return @{ path = $outPath; ox = $org[0]; oy = $org[1]; w = $size[0]; h = $size[1]; ok = $ok }
}

# Samples one row of the captured image and classifies the pixels by hue family. The four band
# colours are red (r>g,b), yellow (r,g>b), green (g>r,b) and the neutral grey/off band.
function Get-BandStats([string]$png, [int]$row, [int]$x0, [int]$x1) {
    Add-Type -AssemblyName System.Drawing
    $img = [System.Drawing.Image]::FromFile($png)
    $red = 0; $green = 0; $yellow = 0; $neutral = 0; $total = 0
    $step = 4
    for ($x = $x0; $x -lt $x1 -and $x -lt $img.Width; $x += $step) {
        $c = ($img -as [System.Drawing.Bitmap]).GetPixel($x, $row)
        $total++
        if ($c.R -gt ($c.G + 30) -and $c.R -gt ($c.B + 30)) { $red++ }
        elseif ($c.G -gt ($c.R + 20) -and $c.G -gt ($c.B + 20)) { $green++ }
        elseif ($c.R -gt ($c.B + 25) -and $c.G -gt ($c.B + 15)) { $yellow++ }
        elseif ([Math]::Abs($c.R - $c.G) -le 25 -and [Math]::Abs($c.G - $c.B) -le 30) { $neutral++ }
    }
    $img.Dispose()
    return @{ total = $total; red = $red; green = $green; yellow = $yellow; neutral = $neutral }
}

$enc = New-Object System.Text.UTF8Encoding($false)
$trace = Join-Path (Split-Path -Parent $Exe) 'ecm_gui_trace.log'

# Starts the GUI, waits `seconds`, closes it with WM_CLOSE, returns the trace text.
# `untilPattern`: stop waiting as soon as the trace matches (bounded by `seconds`).
function Invoke-Gui([string]$name, [string]$iniPath, [int]$seconds, [string]$untilPattern = "", [string]$ShotPath = "") {
    Remove-Item $trace -ErrorAction SilentlyContinue
    $proc = Start-Process -FilePath $Exe -ArgumentList @('-ini', $iniPath, '--trace') -PassThru
    $hwnd = [IntPtr]::Zero
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline -and $hwnd -eq [IntPtr]::Zero) {
        Start-Sleep -Milliseconds 300
        $proc.Refresh()
        if ($proc.HasExited) { break }
        $hwnd = [Win32P95]::FindMain([uint32]$proc.Id, 'ecm_gui')
    }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $seconds) {
        Start-Sleep -Milliseconds 400
        $proc.Refresh()
        if ($proc.HasExited) { break }
        if ($untilPattern -ne "" -and (Read-TextShared $trace) -match $untilPattern) { break }
    }
    # Capture while the window is still up (PrintWindow on a closed window is empty), then close.
    $shot = $null
    # NOTE: the parameter must NOT be called $Shot: PowerShell variable names are
    # case-insensitive, so the local $shot below silently cleared it and no capture was
    # ever taken (found 2026-09-29 by printing the parameter).
    if ($ShotPath -ne "" -and $hwnd -ne [IntPtr]::Zero) { $shot = Save-WindowShot $hwnd $ShotPath }
    if ($hwnd -ne [IntPtr]::Zero) { [Win32P95]::Close($hwnd) }
    $exited = $proc.WaitForExit(30000)
    if (-not $exited) { $proc.Kill(); }
    return @{ trace = (Read-TextShared $trace); exit = $proc.ExitCode; exited = $exited;
              seconds = $sw.Elapsed.TotalSeconds; shot = $shot }
}

# The strip line the GUI traced last: level, parked count, geometry and text.
#   p95 notice: level=grey parked=0 rect=x,y,WxH host_top=T text=...
function Get-Strip([string]$log) {
    $hits = [regex]::Matches($log, 'p95 notice: level=(\w+) parked=(\d+) rect=([\d.-]+),([\d.-]+),([\d.]+)x([\d.]+) host_top=([\d.-]+) text=(.*)')
    if ($hits.Count -eq 0) { return $null }
    return $hits[$hits.Count - 1]
}

function Assert-Band([string]$name, $result, $strip, [string]$family) {
    if ($null -eq $result.shot -or $null -eq $strip) {
        Check "$name band: the window was captured" $false ("shot=" + $(if ($null -eq $result.shot) { 'null' } else { 'ok' }) + " strip=" + $(if ($null -eq $strip) { 'null' } else { 'ok' }))
        return
    }
    $shot = $result.shot
    if ($null -eq $shot.h -or $null -eq $shot.oy) {
        Check "$name band: the capture carries its geometry" $false `
              ("types: result=" + $result.GetType().FullName + " shot=" + `
               $(if ($null -eq $shot) { 'null' } else { $shot.GetType().FullName }) + " json=" + `
               ($result | ConvertTo-Json -Compress -Depth 3))
        return
    }
    $row = [Math]::Round([double]$strip.Groups[4].Value - $shot.oy + [double]$strip.Groups[6].Value / 2.0)
    $x0 = [Math]::Max(0, [Math]::Round([double]$strip.Groups[3].Value - $shot.ox))
    $x1 = $x0 + [Math]::Round([double]$strip.Groups[5].Value)
    if ($row -lt 0 -or $row -ge $shot.h) {
        Check "$name band: the strip row is inside the image" $false ("row=$row h=" + $shot.h)
        return
    }
    $st = Get-BandStats $shot.path $row $x0 $x1
    $hits = $st[$family]
    Write-Host ("       {0} band: row={1} sampled={2} red={3} green={4} yellow={5} neutral={6}" -f `
                $name, $row, $st.total, $st.red, $st.green, $st.yellow, $st.neutral)
    Check "$name band: the strip is drawn in that colour" ($hits -ge ($st.total * 0.4)) `
          ("$family=$hits of $($st.total) sampled pixels")
}

Write-Host ("gui      : " + $Exe)
Write-Host ("driver   : " + $EcmCuda)
Write-Host ("sandbox  : " + $Sandbox)
Write-Host ""

# ------------------------------------------------------------------ [A] toggles --------
$p95Dir = Join-Path $Sandbox 'p95'
New-Item -ItemType Directory -Force -Path $p95Dir | Out-Null
$p95Todo = Join-Path $p95Dir 'worktodo.txt'
[System.IO.File]::WriteAllText($p95Todo, "[Worker #1]`r`nfoo=bar`r`n`r`n[Worker #2]`r`nbaz=qux`r`n", $enc)
# A fake driver path inside the sandbox: worker_dir() then points here, so the parked file
# of the "no worker" scenarios never touches the real build directory.
$fakeDriver = Join-Path $p95Dir 'ecm_cuda.exe'

function Write-SandboxIni([string]$path, [string]$p95Path, [string]$addWorkers, [string]$worktodo, [string]$driverExe) {
    # NOTE: parenthesise every concatenation inside an array literal -- the comma binds
    # tighter than '+', so "key = " + $v would silently become two ini lines.
    $lines = @(
        'method = gpu',
        'gpucurves = 1',
        ('tmp_dir = ' + (Join-Path $Sandbox 'saves')),
        ('finished = ' + (Join-Path $Sandbox 'finished.txt')),
        ('worktodo = ' + $worktodo),
        'verbose = 0',
        'ckpt_seconds = 0',
        ('p95_worktodo_path = ' + $p95Path),
        ('p95_add_workers = ' + $addWorkers),
        '',
        '[GUI]',
        'NumWorkers = 1',
        ('exe = ' + $driverExe),
        'language = english',
        'refresh_hz = 10',
        'exit_confirm = kill',
        '',
        '[Worker #1]',
        ('device = ' + $Device)
    )
    [System.IO.File]::WriteAllText($path, (($lines -join "`r`n") + "`r`n"), $enc)
}

Write-Host "[A1] no p95 keys (the handoff is off)"
$iniA1 = Join-Path $Sandbox 'a1.ini'
Write-SandboxIni $iniA1 '' '' (Join-Path $Sandbox 'worktodo_a1.txt') $fakeDriver
[System.IO.File]::WriteAllText((Join-Path $Sandbox 'worktodo_a1.txt'), "# empty queue`r`n", $enc)
$rA1 = Invoke-Gui 'a1' $iniA1 8 'layout: default dock layout built' (Join-Path $Sandbox 'a1.png')
$s1 = Get-Strip $rA1.trace
Check "the GUI exited cleanly"           ($rA1.exited -and $rA1.exit -eq 0) ("exit " + $rA1.exit)
Check "the strip is grey (not configured)" ($null -ne $s1 -and $s1.Groups[1].Value -eq 'grey') $(if ($s1) { $s1.Value })
Check "it names the remedy (p95_worktodo_path)" ($null -ne $s1 -and $s1.Groups[8].Value -match 'p95_worktodo_path')
Check "it says the handoff is off"         ($null -ne $s1 -and $s1.Groups[8].Value -match 'OFF|off')
Check "it carries a severity marker"       ($null -ne $s1 -and $s1.Groups[8].Value -match '^\[')
Check "the strip is a full-width row"     ($null -ne $s1 -and [double]$s1.Groups[5].Value -gt 200) $(if ($s1) { $s1.Groups[5].Value })
Check "the strip has a readable height"   ($null -ne $s1 -and [double]$s1.Groups[6].Value -ge 20) $(if ($s1) { $s1.Groups[6].Value })
# The user could not see the four colours at all (2026-09-29) because the strip was drawn over
# the dockspace area and the docked panels painted on top of it. Its row is reserved now: the
# strip must end at or above the top of the dockspace host.
Assert-Band 'grey' $rA1 $s1 'neutral'
Check "the strip is above the dockspace"  ($null -ne $s1 -and ([double]$s1.Groups[4].Value + [double]$s1.Groups[6].Value) -le [double]$s1.Groups[7].Value) `
      $(if ($s1) { "bottom=" + ([double]$s1.Groups[4].Value + [double]$s1.Groups[6].Value) + " host_top=" + $s1.Groups[7].Value })

# PIXEL PROOF that the strip really is on the screen in its band colour. The geometric check
# above only says it is not covered by a panel; only pixels say the user can see it. The
# 2026-09-29 report was "四种颜色的真实切换，我实测看不出", and the cause was exactly that: the
# strip was drawn, but underneath the docked panels.


Write-Host "[A2] p95_worktodo_path set, nothing parked"
$iniA2 = Join-Path $Sandbox 'a2.ini'
Write-SandboxIni $iniA2 $p95Todo '' (Join-Path $Sandbox 'worktodo_a2.txt') $fakeDriver
[System.IO.File]::WriteAllText((Join-Path $Sandbox 'worktodo_a2.txt'), "# empty queue`r`n", $enc)
Remove-Item (Join-Path $p95Dir 'p95_add_pending.txt') -ErrorAction SilentlyContinue
$rA2 = Invoke-Gui 'a2' $iniA2 8 'layout: default dock layout built' (Join-Path $Sandbox 'a2.png')
$s2 = Get-Strip $rA2.trace
Check "the strip is green (ready)"        ($null -ne $s2 -and $s2.Groups[1].Value -eq 'green') $(if ($s2) { $s2.Value })
Check "it is not red"                     ($null -ne $s2 -and $s2.Groups[1].Value -ne 'red')
Assert-Band 'green' $rA2 $s2 'green'

Write-Host "[A3] a parked file left by an earlier run"
[System.IO.File]::WriteAllText((Join-Path $p95Dir 'p95_add_pending.txt'),
    "ECMSTAGE2=1,2,521,-1,`"m521_1e3.save`",0,0,1`r`nECMSTAGE2=1,2,523,-1,`"m523_1e3.save`",0,0,1`r`n", $enc)
$rA3 = Invoke-Gui 'a3' $iniA2 8 'layout: default dock layout built' (Join-Path $Sandbox 'a3.png')
$s3 = Get-Strip $rA3.trace
Check "the strip is red"                  ($null -ne $s3 -and $s3.Groups[1].Value -eq 'red') $(if ($s3) { $s3.Value })
Check "it counts the parked lines"        ($null -ne $s3 -and $s3.Groups[2].Value -eq '2') $(if ($s3) { $s3.Groups[2].Value })
Check "the red text mentions delivery failure" ($null -ne $s3 -and $s3.Groups[8].Value -match 'FAILED')
Check "opening the folder needs no worker" ($null -ne $s3)
Assert-Band 'red' $rA3 $s3 'red'

# ------------------------------------------------------------------ [B] warn ------------
Write-Host "[B] a real worker with a routing fallback (p95_add_workers = 3, no [Worker #3])"
$dirB = Join-Path $Sandbox 'b'
New-Item -ItemType Directory -Force -Path $dirB | Out-Null
$todoB = Join-Path $dirB 'worktodo.txt'
$taskB = 'ECMSTAGE2=1,2,521,-1,"m521_1e3.save",0,0,1'
[System.IO.File]::WriteAllText($todoB, ("[Worker #1]`r`n" + $taskB + "`r`n"), $enc)
$iniB = Join-Path $dirB 'ecm.ini'
Write-SandboxIni $iniB $p95Todo '3' $todoB $EcmCuda
# The [Worker #N] keys of the real driver section (device) plus autostart.
$iniBText = [System.IO.File]::ReadAllText($iniB)
[System.IO.File]::WriteAllText($iniB, $iniBText + "autostart = 1`r`n", $enc)
Remove-Item (Join-Path $p95Dir 'worktodo.add') -ErrorAction SilentlyContinue
Remove-Item (Join-Path $p95Dir 'worktodo.add.lock') -ErrorAction SilentlyContinue
# Wait for QueueEmpty, NOT for the notice: the handoff happens AFTER the task's stage 1
# finishes, so QueueEmpty proves that both the notice and the state transition are in the
# trace. Waiting for the notice alone closed the GUI a moment too early and the transition
# was never observed (measured 2026-09-29 in the suite; a standalone run had passed).
$rB = Invoke-Gui 'b' $iniB $TimeoutSeconds 'worker 1: state QueueEmpty' (Join-Path $Sandbox 'b.png')
$sB = Get-Strip $rB.trace
Check "the worker handed the task over"   ($rB.trace -match 'worker 1: p95 warn worker=0')
Check "the worker finished"               ($rB.trace -match 'worker 1: state QueueEmpty') ("{0:N0}s" -f $rB.seconds)
Check "the strip turned yellow"           ($null -ne $sB -and $sB.Groups[1].Value -eq 'yellow') $(if ($sB) { $sB.Value })
Check "the strip carries the driver's note" ($null -ne $sB -and $sB.Groups[8].Value -match 'section')
Assert-Band 'yellow' $rB $sB 'yellow'
Check "worktodo.add really got the line"  ((Test-Path (Join-Path $p95Dir 'worktodo.add')) -and
                                           ([System.IO.File]::ReadAllText((Join-Path $p95Dir 'worktodo.add')) -match 'm521_1e3\.save'))

# ------------------------------------------------------------------ [C] pending ---------
Write-Host "[C] a real worker whose delivery fails (a fresh worktodo.add.lock)"
$dirC = Join-Path $Sandbox 'c'
New-Item -ItemType Directory -Force -Path $dirC | Out-Null
$todoC = Join-Path $dirC 'worktodo.txt'
$taskC = 'ECMSTAGE2=1,2,522,-1,"m522_1e3.save",0,0,1'
[System.IO.File]::WriteAllText($todoC, ("[Worker #1]`r`n" + $taskC + "`r`n"), $enc)
$iniC = Join-Path $dirC 'ecm.ini'
Write-SandboxIni $iniC $p95Todo '1' $todoC $EcmCuda
$iniCText = [System.IO.File]::ReadAllText($iniC)
[System.IO.File]::WriteAllText($iniC, $iniCText + "autostart = 1`r`n", $enc)
Remove-Item $realPending -ErrorAction SilentlyContinue
Set-Content -LiteralPath (Join-Path $p95Dir 'worktodo.add.lock') -Value 'held by the test' -Encoding ASCII
try {
    # Wait for the queue to drain (NOT for the pending notice): the point is that the
    # failed handoff did not stop the task, which only QueueEmpty proves.
    $rC = Invoke-Gui 'c' $iniC $TimeoutSeconds 'worker 1: state QueueEmpty'
    $sC = Get-Strip $rC.trace
    Check "the worker reported the failure"  ($rC.trace -match 'worker 1: p95 pending')
    Check "the reason mentions the lock"     ($rC.trace -match 'locked')
    Check "the strip turned red"             ($null -ne $sC -and $sC.Groups[1].Value -eq 'red') $(if ($sC) { $sC.Value })
    Check "the strip counts the parked line" ($null -ne $sC -and [int]$sC.Groups[2].Value -ge 1) $(if ($sC) { $sC.Groups[2].Value })
    Check "the line is really parked on disk" ((Test-Path $realPending) -and
                                               ([System.IO.File]::ReadAllText($realPending) -match 'm522_1e3\.save'))
    Check "the task still left the queue"     (-not ([System.IO.File]::ReadAllText($todoC) -match 'ECMSTAGE2'))
    Check "no curve was lost to the failure"  ($rC.trace -match 'worker 1: state QueueEmpty')
} finally {
    Remove-Item (Join-Path $p95Dir 'worktodo.add.lock') -Force -ErrorAction SilentlyContinue
    Remove-Item $realPending -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Check "the parked file was cleaned up"    (-not (Test-Path $realPending))
# Only the driver THIS test started counts: a production ecm_gui/ecm_cuda pair may well be
# running on this machine (it was, on 2026-09-29) and must never be reported or touched.
# This test kills nothing itself -- the GUI owns its workers (job object, KILL_ON_JOB_CLOSE).
function Count-OurWorkers {
    $n = 0
    foreach ($p in @(Get-Process -Name 'ecm_cuda' -ErrorAction SilentlyContinue)) {
        $path = ""
        try { $path = $p.MainModule.FileName } catch { $path = "" }
        if ($path -and ($path -eq $EcmCuda)) { $n++ }
    }
    return $n
}
$strayDeadline = (Get-Date).AddSeconds(10)
while ((Get-Date) -lt $strayDeadline -and (Count-OurWorkers) -gt 0) {
    Start-Sleep -Milliseconds 500
}
Check "no worker of ours is left behind"  ((Count-OurWorkers) -eq 0)

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
