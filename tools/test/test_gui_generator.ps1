#Requires -Version 5.1
<#
.SYNOPSIS
    Acceptance test for the worktodo generator (M6 scope A, docs/DEV_ECM_GUI.md 19).

.DESCRIPTION
    The C++ generator in the GUI has to agree with the reference implementation
    tools/ecm_worktodo/ecm.py, so the decisive check is a BYTE-FOR-BYTE comparison of their
    output for the same input and the same options:

      [1] the unit test (ecm_gui_gen_test.exe): --gpu-info parsing, tier pick, effective
          bits, parsing, save names, the pipeline, the mtime-guarded append
      [2] byte-for-byte vs ecm.py: assignments -> ECMSTAGE2= lines with a fixed curve count
          (the case where both must agree exactly)
      [3] the same with duplicates (dedup winner + merged known factors)
      [4] the same with rewrites (--set-b1 / --set-b2 / --set-has-na)
      [5] the recommendation itself: per line, curves = n_blocks/SM x SM count x ipb of the
          tier that line's N would run on, read live from `ecm_cuda.exe --gpu-info` (D4)
      [6] the GUI panel is drawn and traces its state (no clicking needed)

    [2]-[4] pass the SAME explicit curve count to both programs (ecm.py's --gpu-curves is a
    single value for every line), which is exactly what makes them comparable; the
    per-line recommendation is checked separately in [5].

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_gui_generator.ps1
#>
param(
    [string]$Exe = "",
    [string]$GenTest = "",
    [string]$EcmCuda = "",
    [string]$Python = "python",
    [string]$Sandbox = "",
    [int]$Device = 0,
    [int]$Curves = 1920,
    [int]$BlocksPerSm = 2
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Exe) {
    foreach ($cand in @("$repoRoot\build_gui\ecm_gui.exe", "$repoRoot\build_vs18\Release\ecm_gui.exe")) {
        if (Test-Path $cand) { $Exe = $cand; break }
    }
}
if (-not $GenTest) { $GenTest = Join-Path $repoRoot 'build_gui\ecm_gui_gen_test.exe' }
if (-not $EcmCuda) {
    foreach ($cand in @("$repoRoot\build_cuda_cmake\ecm_cuda.exe", "$repoRoot\build_vs18\Release\ecm_cuda.exe")) {
        if (Test-Path $cand) { $EcmCuda = $cand; break }
    }
}
$ecmPy = Join-Path $repoRoot 'tools\ecm_worktodo\ecm.py'
if (-not (Test-Path $Exe)) { Write-Host "FAIL: ecm_gui.exe not found" -ForegroundColor Red; exit 2 }
if (-not (Test-Path $GenTest)) { Write-Host "FAIL: $GenTest not found (build it first)" -ForegroundColor Red; exit 2 }
if (-not (Test-Path $EcmCuda)) { Write-Host "FAIL: ecm_cuda.exe not found" -ForegroundColor Red; exit 2 }
if (-not (Test-Path $ecmPy)) { Write-Host "FAIL: ecm.py not found" -ForegroundColor Red; exit 2 }
if (-not $Sandbox) { $Sandbox = Join-Path $PSScriptRoot '_run\gui_generator' }
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
function Write-Text([string]$path, [string]$text) {
    [System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding($false)))
}

Write-Host ("gui test : " + $GenTest)
Write-Host ("ecm.py   : " + $ecmPy)
Write-Host ("driver   : " + $EcmCuda)
Write-Host ("sandbox  : " + $Sandbox)
Write-Host ""

# ------------------------------------------------------------------ [1] unit test -------
Write-Host "[1] generator unit test"
$unit = (& $GenTest 2>&1 | Out-String)
$m = [regex]::Match($unit, 'passed:\s*(\d+)\s+failed:\s*(\d+)')
Check "the unit test reports a result" ($m.Success) $unit
if ($m.Success) {
    Check "no unit check failed" ([int]$m.Groups[2].Value -eq 0) `
        ("failed=" + $m.Groups[2].Value + " :: " + (($unit -split "`r?`n" | Where-Object { $_ -match 'FAIL' }) -join ' | '))
    Write-Host ("       passed=" + $m.Groups[1].Value)
}

# ------------------------------------------------------------------ helpers -------------
# The sample keeps every line in ONE tier so a single --gpu-curves value is comparable.
$sampleMain = @(
    '# PrimeNet manual assignments',
    'ECM2=AID0000000000000000000000000001,1,2,561,-1,110e6,0,120,"3,5"',
    'ECM2=1,2,577,-1,3e6,0,64',
    'ECM2=FFT2=1024,1,2,601,-1,11000000,0,8',
    'ECM2=1,2,619,-1,11e6,0,16,"7"'
)
$sampleDup = @(
    'ECM2=1,2,521,-1,1e5,0,8',
    'ECM2=1,2,521,-1,1e6,0,8,"11"',
    'ECM2=1,2,521,-1,5e5,0,8,"13"',
    'ECM2=1,2,700,-1,1e5,0,8'
)

function Compare-WithEcmPy([string]$name, [string[]]$lines, [string[]]$extraEcmPy, [string[]]$extraGen) {
    $input = Join-Path $Sandbox ($name + '.txt')
    $py = Join-Path $Sandbox ($name + '.py.txt')
    $mine = Join-Path $Sandbox ($name + '.gen.txt')
    Write-Text $input (($lines -join "`r`n") + "`r`n")
    Remove-Item $py, $mine -Force -ErrorAction SilentlyContinue

    # NOTE: no PowerShell switches here -- this is the python interpreter, not powershell.
    $pyArgs = @($ecmPy, '--input', $input,
                '--out-ecmstage2', $py, '--save-pattern', 'm{n}_{b1}.save',
                '--gpu-curves', "$Curves", '--skip-curves', '0') + $extraEcmPy
    & $Python @pyArgs *>&1 | Out-Null
    $pyExit = $LASTEXITCODE

    $genArgs = @('--emit', $input, '--write', $mine, '--workers', '0=0', '--curves', "$Curves",
                 '--save-pattern', 'm{n}_{b1}.save') + $extraGen
    & $GenTest @genArgs *>&1 | Out-Null
    $genExit = $LASTEXITCODE

    Check "${name}: ecm.py succeeded" ($pyExit -eq 0) ("exit " + $pyExit)
    Check "${name}: the generator succeeded" ($genExit -eq 0) ("exit " + $genExit)
    if ($pyExit -ne 0 -or $genExit -ne 0) { return }

    $a = [System.IO.File]::ReadAllBytes($py)
    $b = [System.IO.File]::ReadAllBytes($mine)
    $same = ($a.Length -eq $b.Length)
    if ($same) {
        for ($i = 0; $i -lt $a.Length; $i++) { if ($a[$i] -ne $b[$i]) { $same = $false; break } }
    }
    if ($same) {
        Check "${name}: byte-for-byte identical to ecm.py" $true
    } else {
        $at = [System.IO.File]::ReadAllText($py)
        $bt = [System.IO.File]::ReadAllText($mine)
        $al = $at -split "`r`n"; $bl = $bt -split "`r`n"
        $where = "(same length: $($a.Length) vs $($b.Length))"
        for ($i = 0; $i -lt [Math]::Max($al.Count, $bl.Count); $i++) {
            $x = if ($i -lt $al.Count) { $al[$i] } else { '<missing>' }
            $y = if ($i -lt $bl.Count) { $bl[$i] } else { '<missing>' }
            if ($x -ne $y) { $where = "line $($i + 1): ecm.py=[$x] gen=[$y]"; break }
        }
        Check "${name}: byte-for-byte identical to ecm.py" $false $where
    }
    # Show one emitted line so a failure is readable (PowerShell 5.1 has no ?: operator).
    $pyLines = @(Get-Content -LiteralPath $py -ErrorAction SilentlyContinue)
    if ($pyLines.Count -gt 1) { Write-Host ("       first emitted line: " + $pyLines[1]) }
}

# ------------------------------------------------------------------ [2] parity ----------
Write-Host "[2] byte-for-byte vs ecm.py (plain assignments, sort by n)"
Compare-WithEcmPy 'main' $sampleMain @('--sort-by', 'n') @('--sort-by', 'n')
Compare-WithEcmPy 'sortb1' $sampleMain @('--sort-by', 'b1') @('--sort-by', 'b1')

# ------------------------------------------------------------------ [3] dedup -----------
Write-Host "[3] byte-for-byte vs ecm.py (duplicates -> dedup + merged factors)"
Compare-WithEcmPy 'dedup' $sampleDup @('--sort-by', 'n') @('--sort-by', 'n')

# ------------------------------------------------------------------ [4] rewrites --------
Write-Host "[4] byte-for-byte vs ecm.py (rewrites)"
Compare-WithEcmPy 'rewrite' $sampleMain @('--sort-by', 'n', '--set-b1', '2e6', '--set-b2', '5e9') `
                                     @('--sort-by', 'n', '--set-b1', '2e6', '--set-b2', '5e9')
Compare-WithEcmPy 'hasna' $sampleDup @('--sort-by', 'n', '--set-has-na') @('--sort-by', 'n', '--set-has-na')

# ------------------------------------------------------------------ [5] recommendation --
Write-Host "[5] per-line recommendation from --gpu-info"
$gpuOut = (& $EcmCuda --gpu-info -d $Device 2>&1 | Out-String)
$sm = [int]([regex]::Match($gpuOut, '(?m)^sm_count=(\d+)').Groups[1].Value)
$carry = [int]([regex]::Match($gpuOut, '(?m)^carry_bits=(\d+)').Groups[1].Value)
$tiers = @()
foreach ($tm in [regex]::Matches($gpuOut, '(?m)^tier bits=(\d+) tpb=(\d+) tpi=(\d+) ipb=(\d+)')) {
    $tiers += [pscustomobject]@{ bits = [int]$tm.Groups[1].Value; ipb = [int]$tm.Groups[4].Value }
}
Check "the driver reported its tiers" ($sm -gt 0 -and $tiers.Count -gt 0) $gpuOut
function Expected-Curves([int]$n) {
    foreach ($t in $tiers) { if ($t.bits -ge ($n + $carry)) { return $BlocksPerSm * $sm * $t.ipb } }
    return 0
}
$recInput = Join-Path $Sandbox 'rec.txt'
Write-Text $recInput ("ECM2=1,2,101,-1,1e5`r`nECM2=1,2,521,-1,1e5`r`nECM2=1,2,1019,-1,1e5`r`n")
$rec = (& $GenTest --emit $recInput --workers '1=0' --sort-by 'n' 2>&1 | Out-String)
$recLines = @($rec -split "`r?`n" | Where-Object { $_ -match '^ECMSTAGE2=' })
Check "three lines were emitted" ($recLines.Count -eq 3) ($recLines -join ' | ')
# NOTE: every call needs its own parentheses -- inside an array literal the comma binds
# tighter than the call, so `Expected-Curves 101, (...)` would pass two arguments.
$expects = @((Expected-Curves 101), (Expected-Curves 521), (Expected-Curves 1019))
Write-Host ("       expected curves: " + ($expects -join ', '))
$ok = $true
for ($i = 0; $i -lt [Math]::Min(3, $recLines.Count); $i++) {
    # ECMSTAGE2=...,<save>,<B2>,<skip>,<curves>[,<factors>]
    $fields = $recLines[$i].Substring(10) -split ','
    $curvesOfLine = [int]$fields[7]
    if ($curvesOfLine -ne $expects[$i]) {
        $ok = $false
        Write-Host ("       line $($i + 1): curves=$curvesOfLine expected=$($expects[$i])")
    }
}
Check "each line carries n_blocks/SM x sm_count x ipb of its own tier" $ok
Check "the tiers differ (so the test is meaningful)" ($expects[0] -ne $expects[2]) ($expects -join ',')

# ------------------------------------------------------------------ [6] GUI panel -------
Write-Host "[6] the GUI panel exists and traces its state"
$iniPath = Join-Path $Sandbox 'gui.ini'
$p95Todo = Join-Path $Sandbox 'p95_worktodo.txt'
Write-Text $p95Todo "[Worker #1]`r`nfoo=bar`r`n"
$todo = Join-Path $Sandbox 'gui_worktodo.txt'
Write-Text $todo "# empty queue (the generator appends here)`r`n"
# NOTE: parenthesise every concatenation inside an array literal -- the comma binds tighter
# than '+', so "key = " + $v would silently become two ini lines.
Write-Text $iniPath ((@(
    'method = gpu',
    ('tmp_dir = ' + $Sandbox),
    ('finished = ' + (Join-Path $Sandbox 'finished.txt')),
    ('worktodo = ' + $todo),
    ('p95_worktodo_path = ' + $p95Todo),
    'p95_add_workers =',
    '',
    '[GUI]',
    'NumWorkers = 1',
    ('exe = ' + $EcmCuda),
    'language = english',
    'refresh_hz = 10',
    'exit_confirm = kill',
    # Open the GUI straight on the generator: a dock tab that is not selected is skipped by
    # ImGui and reports NO geometry, so the control checks below need it visible.
    'start_tab = gen',
    '',
    '[Worker #1]',
    ('device = ' + $Device)
) -join "`r`n") + "`r`n")
$trace = Join-Path (Split-Path -Parent $Exe) 'ecm_gui_trace.log'
Remove-Item $trace -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $Exe -ArgumentList @('-ini', $iniPath, '--trace') -PassThru
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$log = ""
while ($sw.Elapsed.TotalSeconds -lt 12) {
    Start-Sleep -Milliseconds 500
    $proc.Refresh()
    if ($proc.HasExited) { break }
    $log = Read-TextShared $trace
    if ($log -match 'gen: panel preview=') { break }
}
# The window is opened for real; close it with WM_CLOSE via the class the GUI registers.
if (-not ("Win32Gen" -as [type])) {
    Add-Type @'
using System; using System.Runtime.InteropServices; using System.Text;
public static class Win32Gen {
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
$hwnd = [Win32Gen]::Find([uint32]$proc.Id, 'ecm_gui')
if ($hwnd -ne [IntPtr]::Zero) { [Win32Gen]::Close($hwnd) }
if (-not $proc.WaitForExit(30000)) { $proc.Kill() }
$log = Read-TextShared $trace
Check "the GUI drew the generator panel" ($log -match 'draw: gen panel ok')
Check "the panel traced its state" ($log -match 'gen: panel preview=0 lines=0 segments=0')
Check "the panel names the target file" ($log -match 'gen: panel .*target=.*gui_worktodo\.txt')
Check "[GUI] start_tab selected the generator" ($log -match 'layout: start_tab is visible: ###gen')
# The two checks the user's report is about (2026-09-29: "block/sm 输入框没有宽度，无法显示数字"):
# the box must exist at a usable width, and the WHOLE box must be the editable area. ImGui's
# InputInt reserves 2*GetFrameHeight() inside the item for +/- step buttons, which at 150 % DPI
# left ~24 px of an 80 px box -- `blocks_edit_w` is what caught that.
$box = [regex]::Match($log, 'gen: panel [^\r\n]*blocks_box_w=([\d.]+) blocks_edit_w=([\d.]+) blocks_per_sm=(\d+)')
Check "the panel traces its control geometry" ($box.Success) "no 'blocks_box_w=...' line in the trace"
if ($box.Success) {
    $boxW = [double]$box.Groups[1].Value
    $editW = [double]$box.Groups[2].Value
    $value = [int]$box.Groups[3].Value
    Write-Host ("       blocks/SM box: frame={0} px, editable={1} px, value={2}" -f $boxW, $editW, $value)
    Check "the blocks/SM box has a real width" ($boxW -ge 60) ("box_w=" + $boxW)
    Check "the whole box is editable (no step buttons)" ($editW -ge 60) ("edit_w=" + $editW)
    Check "the box holds the default value 2" ($value -eq 2) ("value=" + $value)
    # 2 digits (up to 64) must fit next to the frame padding, on any DPI.
    Check "two digits fit in the editable width" ($editW -ge (2 * 12)) ("edit_w=" + $editW)
}
# The ini names an ABSOLUTE worktodo path here: the panel must show exactly that file, not the
# worker directory prefixed to it (which is what it did before: "…\sandbox\D:\…\worktodo.txt").
Check "the target path is not doubled" ($log -match ('gen: panel .*target=' + [regex]::Escape($todo) + '\b'))
Check "the GUI exited cleanly" ($proc.HasExited -and $proc.ExitCode -eq 0) ("exit " + $proc.ExitCode)

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
