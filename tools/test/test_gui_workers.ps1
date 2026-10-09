#Requires -Version 5.1
<#
.SYNOPSIS
    M2 acceptance: the GUI supervises worker processes (docs/usage/GUI.md).

.DESCRIPTION
    Starts the real ecm_gui with a sandbox ecm.ini that points [GUI] exe= at the fake
    worker, autostarts two workers --
        Worker #1: --scenario ok          (prints a scripted run, exits 0 "queue done")
        Worker #2: --scenario crash-once  (exit 7 the first time, then like ok)
    -- waits, closes the window through WM_CLOSE, and then reads the --trace file to
    assert the whole lifecycle: autostart, Running -> QueueEmpty for #1, a crash +
    restart #1 for #2, and that closing the GUI stopped both.

    Everything asserted here is observable in the trace, so the test needs no
    screenshots and no GPU.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_gui_workers.ps1
#>
param(
    [string]$Exe = "",
    [string]$Fake = "",
    [string]$Sandbox = "",
    [int]$RunSeconds = 12
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Exe) {
    foreach ($cand in @("$repoRoot\build_gui\ecm_gui.exe",
                        "$repoRoot\build_vs18\Release\ecm_gui.exe")) {
        if (Test-Path $cand) { $Exe = $cand; break }
    }
}
if (-not $Fake) { $Fake = Join-Path (Split-Path -Parent $Exe) 'ecm_gui_fake_worker.exe' }
if (-not (Test-Path $Exe)) { Write-Host "FAIL: ecm_gui.exe not found" -ForegroundColor Red; exit 2 }
if (-not (Test-Path $Fake)) { Write-Host "FAIL: ecm_gui_fake_worker.exe not found (build target ecm_gui_fake_worker)" -ForegroundColor Red; exit 2 }
if (-not $Sandbox) { $Sandbox = Join-Path $PSScriptRoot '_run\gui_workers' }
if (Test-Path $Sandbox) { Remove-Item $Sandbox -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Sandbox | Out-Null

$script:pass = 0
$script:fail = 0
function Check([string]$name, [bool]$ok, [string]$detail = "") {
    if ($ok) { $script:pass++; Write-Host ("  [ok]   " + $name) }
    else {
        $script:fail++
        Write-Host ("  [FAIL] " + $name + $(if ($detail) { " -- " + $detail } else { "" })) -ForegroundColor Red
    }
}

if (-not ("Win32GuiWorkers" -as [type])) {
    Add-Type @'
using System; using System.Runtime.InteropServices; using System.Text;
public static class Win32GuiWorkers {
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
$marker = Join-Path $Sandbox 'crashed.once.marker'
$enc = New-Object System.Text.UTF8Encoding($false)
$iniLines = @(
    '# M2 sandbox: two fake workers, one of them crashes once',
    'method = gpu',
    'device = 0',
    '',
    '[GUI]',
    'NumWorkers = 5',
    ('exe = ' + $Fake),
    'language = chineseSimplified',
    'refresh_hz = 15',
    # Closing the window with workers running asks by default (App::request_close). This test
    # performs a scripted close, so use the modal-free policy -- and a short checkpoint wait,
    # because fake workers never write one (the real-driver case is covered by
    # test_gui_exit_checkpoint.ps1).
    'exit_confirm = stop',
    'graceful_stop_ms = 3000',
    '',
    '[Worker #1]',
    'name = steady',
    'autostart = 1',
    # The progress lines must be SPACED OUT (`--delay-ms`): printed back to back they are
    # drained in one poll and only the last one is seen, so the chart would hold a single
    # sample. Six spaced lines give the pane a real series to draw.
    'extra_args = --scenario ok --lines 6 --delay-ms 700',
    '',
    '[Worker #2]',
    'name = flaky',
    'device = 1',
    'autostart = 1',
    ('extra_args = --scenario crash-once --marker ' + $marker),
    '',
    '[Worker #3]',
    'name = hanging',
    'device = 1',
    'autostart = 1',
    'extra_args = --scenario hang',
    '',
    '# A worker executable that predates the --worker support (D1/D2): the GUI must
    # diagnose it instead of only restarting it (user report, 2026-09-28).',
    '[Worker #4]',
    'name = stale',
    'device = 1',
    'autostart = 1',
    'extra_args = --scenario stale-driver',
    '',
    '# A RESUMED run (the real user case): the driver prints the resume percentage and then',
    '# NO progress line for a long time, because its redirected progress schedule is gated on',
    '# the batch counter restored from the checkpoint. The GUI must seed the bar from it.',
    '[Worker #5]',
    'name = resumed',
    'device = 1',
    'autostart = 1',
    'extra_args = --scenario resumed --pct 23.8'
)
[System.IO.File]::WriteAllText($iniPath, (($iniLines -join "`r`n") + "`r`n"), $enc)

$trace = Join-Path (Split-Path -Parent $Exe) 'ecm_gui_trace.log'
Remove-Item $trace -ErrorAction SilentlyContinue

Write-Host ("exe     : " + $Exe)
Write-Host ("fake    : " + $Fake)
Write-Host ("sandbox : " + $Sandbox)
Write-Host ""
Write-Host ("[1] start the GUI and let it supervise for {0}s" -f $RunSeconds)

$proc = Start-Process -FilePath $Exe -ArgumentList @('-ini', $iniPath, '--trace') -PassThru
$hwnd = [IntPtr]::Zero
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline -and $hwnd -eq [IntPtr]::Zero) {
    Start-Sleep -Milliseconds 400
    $proc.Refresh()
    if ($proc.HasExited) { break }
    $hwnd = [Win32GuiWorkers]::Find([uint32]$proc.Id, 'ecm_gui')
}
Check "GUI window appeared" ($hwnd -ne [IntPtr]::Zero)
Start-Sleep -Seconds $RunSeconds

Write-Host "[2] close it (WM_CLOSE) - shutdown must stop the workers"
if ($hwnd -ne [IntPtr]::Zero) { [Win32GuiWorkers]::Close($hwnd) }
if (-not $proc.WaitForExit(30000)) { $proc.Kill(); Check "GUI exited" $false "had to kill it" }
else { Check "GUI exited cleanly" ($proc.ExitCode -eq 0) ("exit code " + $proc.ExitCode) }

Write-Host "[3] the trace shows the supervised lifecycle"
$log = ''
if (Test-Path $trace) { $log = [System.IO.File]::ReadAllText($trace) }
Check "trace file written" ($log.Length -gt 0) $trace
Check "worker 1 autostarted"     ($log -match 'worker 1: autostart pid=\d+')
Check "worker 2 autostarted"     ($log -match 'worker 2: autostart pid=\d+')
Check "command line uses --worker N" ($log -match '--worker 1') 
Check "worker 1 reached Running" ($log -match 'worker 1: state Running')
Check "worker 1 ended QueueEmpty" ($log -match 'worker 1: state QueueEmpty')
Check "worker 2 crashed -> Restarting" ($log -match 'worker 2: state Restarting')
Check "worker 2 restarted exactly once" (([regex]::Matches($log, 'worker 2: restart #\d+')).Count -eq 1)
Check "worker 2 then finished its queue" ($log -match 'worker 2: state QueueEmpty')
Check "worker 3 is still running when the GUI closes" ($log -match 'worker 3: state Running')
# Closing asks for a checkpoint first (user requirement, docs/usage/GUI.md 5.6). This test
# runs with exit_confirm = stop (no modal) and a 3 s checkpoint wait, and its fake workers
# never write a checkpoint -- so the expected trace is: request -> timeout -> stop, and
# nothing crash-like in between or after.
Check "the exit flow asks for a checkpoint before stopping" `
      ($log -match 'worker 3: graceful stop requested \(waiting for a checkpoint') `
      "no 'graceful stop requested' line for worker 3"
Check "the stop line says whether a checkpoint was written" `
      ($log -match 'worker 3: stopping \(checkpoint (written|NOT written)') `
      "no 'stopping (checkpoint ...)' line for worker 3"
$exitAt = $log.LastIndexOf('worker 3: graceful stop requested')
if ($exitAt -ge 0) {
    $afterExit = $log.Substring($exitAt)
    Check "no crash/restart reporting during the exit flow" `
          ($afterExit -notmatch 'state Error|state Restarting|restart #') `
          "found error/restart lines after the exit flow started"
}
# The 4th worker is a driver too old to understand --worker: the GUI must name the cause
# (this is the user-reported "Start fails, all settings look right" case).
Check "worker 4 autostarted"     ($log -match 'worker 4: autostart pid=\d+')
Check "worker 4 is diagnosed as a too-old driver" `
      ($log -match "worker 4: DIAGNOSIS: [^\r\n]*--worker[^\r\n]*older than the D1/D2") `
      "no DIAGNOSIS line for worker 4"
Check "worker 4 stops after the crash breaker trips" ($log -match 'worker 4: state Error')
Check "worker 5 sees the resume percentage and seeds the bar from it" `
      ($log -match 'worker 5: progress seed=23\.8 \(from checkpoint\)') `
      "no 'progress seed=... (from checkpoint)' line for worker 5"
Check "worker 5 still reports no progress line of its own" `
      ($log -notmatch 'worker 5: progress pct=') "worker 5 unexpectedly got a progress line"

Write-Host "[3b] the Workers table keeps the progress bar visible (user report: it was cut off)"
# The task line used to live in a table column: an ~80 character worktodo line auto-sized
# that column, so progress/speed/ETA were pushed past the panel's right edge and the user
# saw no progress bar and no ETA at all. Now the task gets its own row, and the trace
# reports the measured geometry.
$tbls = [regex]::Matches($log, "table: workers right=([\d.]+) left=([\d.]+) progress_x=([\d.]+) progress_w=([\d.]+) state_x=([\d.]+) name_x=([\d.]+) speed_x=([\d.]+) eta_x=([\d.]+) actions_x=([\d.]+) task_wrap_w=([\d.]+) task_lines=(\d+) rows=(\d+) fits=(\d)")
Check "the trace reports the Workers table geometry" ($tbls.Count -gt 0)
if ($tbls.Count -gt 0) {
    $tbl = $tbls[$tbls.Count - 1]      # the last one: after the dock layout settled
    $right = [double]$tbl.Groups[1].Value
    $px = [double]$tbl.Groups[3].Value
    $pw = [double]$tbl.Groups[4].Value
    $speedX = [double]$tbl.Groups[7].Value
    $etaX = [double]$tbl.Groups[8].Value
    $taskW = [double]$tbl.Groups[10].Value
    Write-Host ("       right=" + $right + " progress=" + $px + "+" + $pw +
                " speed_x=" + $speedX + " eta_x=" + $etaX + " task_wrap_w=" + $taskW +
                " task_lines=" + $tbl.Groups[11].Value + " rows=" + $tbl.Groups[12].Value)
    Check "the progress bar has real width" ($pw -ge 60) ("progress_w=" + $pw)
    Check "the progress bar ends inside the table" (($px + $pw) -le ($right + 2)) `
          ("x+w=" + ($px + $pw) + " right=" + $right)
    Check "the speed column follows the progress bar" ($speedX -ge ($px + $pw)) `
          ("speed_x=" + $speedX + " bar ends at " + ($px + $pw))
    # This is the user's complaint: the ETA column used to be pushed past the right edge.
    Check "the ETA column is inside the panel" ($etaX -lt $right) ("eta_x=" + $etaX + " right=" + $right)
    Check "the layout check self-reports fits=1" ($tbl.Groups[13].Value -eq '1')
    # The task line must span the row: a narrow first column (28 px) would show only the
    # first couple of characters (user report 2026-09-28).
    Check "the task line spans the row, not one column" ($taskW -ge 300) ("task_wrap_w=" + $taskW)
    Check "the task lines are drawn on their own rows" ([int]$tbl.Groups[11].Value -ge 1) `
          ("task_lines=" + $tbl.Groups[11].Value)
    # One row per worker plus one task row per worker that reported a task.
    Check "the table has the worker rows" ([int]$tbl.Groups[12].Value -eq 5) `
          ("rows=" + $tbl.Groups[12].Value)
}

Write-Host "[3c] the worker output pane charts SECONDS PER CURVE (and no progress-% chart)"
# User request 2026-09-29: "worker 日志窗口删去进度%，曲线/秒 改为 秒/曲线".
# The charts are traced (`plot: <id> label="…" n=… lo=… hi=… last=…`) because a chart has no
# other machine-readable footprint. The fake worker's progress line says "~1.20 s/curve", so
# the plotted value must be ~1.2 -- the OLD charted quantity was 1/s_per_curve = ~0.83, which
# is how this assertion tells the two apart.
$plots = [regex]::Matches($log, 'plot: (\S+) label="([^"]*)" n=(\d+) lo=([\d.-]+) hi=([\d.-]+) last=([\d.-]+)')
Check "the output pane traces its chart" ($plots.Count -gt 0)
$speed = $null
# Every worker pane is only DRAWN while its worker is the selected one, so the LAST trace for a
# worker chart may be a 1-sample snapshot from the moment the pane became visible. Take the trace
# with the most samples: that is the one that proves a real series is plotted (same unit for all
# workers, so which worker it came from does not matter).
foreach ($m in $plots) {
    if ($m.Groups[1].Value -match '^worker\d+/s_per_curve$') {
        if ($null -eq $speed -or [int]$m.Groups[3].Value -gt [int]$speed.Groups[3].Value) { $speed = $m }
    }
}
Check "a per-worker speed chart exists" ($null -ne $speed)
if ($null -ne $speed) {
    Write-Host ("       " + $speed.Value)
    Check "it has samples" ([int]$speed.Groups[3].Value -ge 2) ("n=" + $speed.Groups[3].Value)
    # The label is localized: English "s/curve (lower is better)", Chinese "秒/曲线（越小越快）".
    # The Chinese check uses \uXXXX escapes because this .ps1 is ASCII-only (a Chinese literal
    # in a BOM-less .ps1 is decoded as GBK and then never matches -- a documented trap).
    Check "it is labelled seconds/curve" `
          (($speed.Groups[2].Value -match 's/curve') -or ($speed.Groups[2].Value -match '\u79d2/\u66f2\u7ebf')) `
          $speed.Groups[2].Value
    Check "and NOT the old curves/s label" ($speed.Groups[2].Value -notmatch 'curves/s')
    # The unit conversion: s/curve ~1.2, not curves/s ~0.83.
    $last = [double]$speed.Groups[6].Value
    # Discriminating bound: the fake worker reports "~1.20 s/curve", so the series must be
    # ~1.2. The OLD charted quantity was curves/s = 1/1.20 = ~0.83, which is < 1.0 and
    # therefore fails this check.
    Check "the plotted value is s/curve (not curves/s)" ($last -ge 1.0 -and $last -le 100.0) `
          ("last=" + $last + " (curves/s would be ~0.83, and a big B1 would be >> 1)")
    Check "its range brackets the value" (([double]$speed.Groups[4].Value -le $last) -and ([double]$speed.Groups[5].Value -ge $last))
}
Check "no progress-% chart is drawn any more" `
      (@($plots | Where-Object { $_.Groups[1].Value -match 'pct|progress' }).Count -eq 0)
Check "and no chart is labelled with a progress percentage" `
      (@($plots | Where-Object { $_.Groups[2].Value -match 'progress %|进度 %' }).Count -eq 0)

Write-Host "[3d] every chart card keeps its three bands disjoint (150 % DPI regression guard)"
# The card is "headline row + curve band + min/max row". The height used to be a constant, while
# the text rows scale with the font -- at 150 % DPI (font 22.5 px, line height ~31 px) the two
# text rows overlapped each other and covered the curve. The trace now reports the geometry, so
# this is asserted instead of eyeballed.
$geoms = @{}
foreach ($m in [regex]::Matches($log, 'plot: (\S+) label="[^"]*" n=\d+ lo=[\d.-]+ hi=[\d.-]+ last=[\d.-]+(?: ref=[\d.-]+)? h=(\d+) line_h=(\d+) band_top=(\d+) band_h=(\d+)')) {
    $geoms[$m.Groups[1].Value] = $m      # last one wins ([regex]::Match returns the first)
}
Check "the trace reports chart geometry" ($geoms.Count -gt 0) ("charts=" + (($geoms.Keys | Sort-Object) -join ','))
foreach ($id in ($geoms.Keys | Sort-Object)) {
    $g = $geoms[$id]
    $h = [double]$g.Groups[2].Value
    $lh = [double]$g.Groups[3].Value
    $bt = [double]$g.Groups[4].Value
    $bh = [double]$g.Groups[5].Value
    Write-Host ("       {0}: h={1} line_h={2} band={3}..{4}" -f $id, $h, $lh, $bt, ($bt + $bh))
    Check ("{0}: the curve band starts below the headline row" -f $id) ($bt -ge $lh + 2) `
          ("band_top=" + $bt + " line_h=" + $lh)
    Check ("{0}: the curve band ends above the min/max row" -f $id) (($bt + $bh) -le ($h - $lh)) `
          ("band_end=" + ($bt + $bh) + " min/max row starts at " + ($h - $lh))
    Check ("{0}: the curve band is at least 24 px tall" -f $id) ($bh -ge 24) ("band_h=" + $bh)
}

Write-Host "[4] no worker processes are left behind"
Start-Sleep -Seconds 2
$stray = @(Get-Process -Name 'ecm_gui_fake_worker' -ErrorAction SilentlyContinue)
Check "no stray fake workers (job objects killed them)" ($stray.Count -eq 0) (("still running: " + (($stray | ForEach-Object { $_.Id }) -join ',')))

Write-Host "[5] the ini still holds the driver keys plus the [GUI] layout"
$after = [System.IO.File]::ReadAllText($iniPath)
Check "device kept"        ($after -match '(?m)^device\s*=\s*0\s*$')
Check "worker name kept"   ($after -match '(?m)^name\s*=\s*steady\s*$')
Check "extra_args kept"    ($after -match '(?m)^extra_args\s*=\s*--scenario ok --lines 6 --delay-ms 700\s*$')
Check "autostart kept"     ($after -match '(?m)^autostart\s*=\s*1\s*$')
Check "window= written"    ($after -match '(?m)^window\s*=\s*-?\d+,-?\d+,\d+,\d+')
Check "dock_layout written" ($after -match '(?m)^dock_layout\s*=\s*\S')

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
