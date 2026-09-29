#Requires -Version 5.1
<#
.SYNOPSIS
    M4 acceptance: the GPU panel's data source (NVML) and its degradation path
    (docs/DEV_ECM_GUI.md section 14, milestone M4).

.DESCRIPTION
    Three layers, each independently checkable:

      1. ecm_gui.exe --gpu-selftest   : loads NVML, samples both the one-shot and the
         threaded paths, sanity-checks every field and cross-checks the values against
         nvidia-smi (see gpu_selftest.cpp for why the comparison is interleaved and
         range-based).
      2. the real GUI with --trace    : must report "gpu: nvml ok" plus one line per
         device, and see the same device count/names as nvidia-smi.
      3. ECM_GUI_NVML=<bogus>         : the GUI must still start, say NVML is
         unavailable, and exit cleanly -- a machine without a usable NVML must never be
         blocked by the GPU panel.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_gui_gpu.ps1
#>
param(
    [string]$Exe = "",
    [string]$Sandbox = ""
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Exe) {
    foreach ($cand in @("$repoRoot\build_gui\ecm_gui.exe", "$repoRoot\build_vs18\Release\ecm_gui.exe")) {
        if (Test-Path $cand) { $Exe = $cand; break }
    }
}
if (-not (Test-Path $Exe)) { Write-Host "FAIL: ecm_gui.exe not found" -ForegroundColor Red; exit 2 }
if (-not $Sandbox) { $Sandbox = Join-Path $PSScriptRoot '_run\gui_gpu' }
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
        $t = $sr.ReadToEnd(); $sr.Close(); $fs.Close(); return $t
    } catch { return "" }
}
if (-not ("Win32GuiGpu" -as [type])) {
    Add-Type @'
using System; using System.Runtime.InteropServices; using System.Text;
public static class Win32GuiGpu {
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

Write-Host ("exe     : " + $Exe)
Write-Host ("sandbox : " + $Sandbox)
Write-Host ""

# ------------------------------------------------------- 1. headless monitor ----
Write-Host "[1] --gpu-selftest (NVML load, fields, thread, nvidia-smi cross-check)"
$out = (& cmd /c ('"{0}" --gpu-selftest 2>&1' -f $Exe) | Out-String)
$code = $LASTEXITCODE
$logPath = Join-Path (Split-Path -Parent $Exe) 'ecm_gui_gpu_selftest.log'
$log = Read-TextShared $logPath
$summary = ($log -split "`r?`n" | Where-Object { $_ -match 'passed:' } | Select-Object -Last 1)
Check "self-test exited 0" ($code -eq 0) ("exit code " + $code)
Check "no failed check in the report" ($log -match 'failed: 0') $summary
if ($log -notmatch 'failed: 0') {
    Write-Host "  --- failing checks from the log ---"
    foreach ($l in ($log -split "`r?`n" | Where-Object { $_ -match '\[FAIL\]' })) { Write-Host ("  " + $l) }
    Write-Host "  --- tail of the log ---"
    foreach ($l in ($log -split "`r?`n" | Select-Object -Last 14)) { Write-Host ("  " + $l) }
}
Check "NVML was loaded" ($log -match 'nvml: \S+')
Check "at least one device sampled" ($log -match 'gpu 0: ')
Check "nvidia-smi cross-check passed" ($log -match 'matches nvidia-smi')

# ---------------------------------------------------- 2. the GUI reports it ----
Write-Host "[2] the GUI picks the monitor up and traces the devices"
$ini = Join-Path $Sandbox 'ecm.ini'
$enc = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($ini, ("[GUI]`r`nNumWorkers = 1`r`nlanguage = english`r`n"), $enc)
$trace = Join-Path (Split-Path -Parent $Exe) 'ecm_gui_trace.log'
Remove-Item $trace -ErrorAction SilentlyContinue

$proc = Start-Process -FilePath $Exe -ArgumentList @('-ini', $ini, '--trace') -PassThru
$hwnd = [IntPtr]::Zero
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline -and $hwnd -eq [IntPtr]::Zero) {
    Start-Sleep -Milliseconds 400
    $proc.Refresh()
    if ($proc.HasExited) { break }
    $hwnd = [Win32GuiGpu]::Find([uint32]$proc.Id, 'ecm_gui')
}
Check "GUI window appeared" ($hwnd -ne [IntPtr]::Zero)
Start-Sleep -Seconds 2
$t = Read-TextShared $trace
Check "trace says NVML is up" ($t -match 'gpu: nvml ok')
Check "trace lists device 0" ($t -match 'gpu 0: ')
# The GPU panel itself must draw without crashing: a %s/%u mismatch in one of its Text()
# calls crashed the whole GUI with 0xC0000005 as soon as NVML reported a non-zero VRAM
# clock (2026-09-29), and every check above still passed -- the process simply died later.
Check "the GPU panel completed and frame 1 was rendered" `
      (($t -match 'draw: gpu panel ok') -and ($t -match 'draw: exit modal ok') -and ($t -match 'frame 1 rendered')) `
      "no first-frame stage trace: the GUI crashed while drawing"
Check "the GUI was still alive a second later" (-not $proc.HasExited)
$smiNames = @(& nvidia-smi --query-gpu=name --format=csv,noheader)
$smiCount = $smiNames.Count
$traceCount = ([regex]::Matches($t, '(?m)^\[[^\]]+\] gpu \d+: ')).Count
Check ("trace device count matches nvidia-smi (" + $smiCount + ")") ($traceCount -eq $smiCount) ("trace=" + $traceCount)
foreach ($n in $smiNames) {
    $short = $n.Trim()
    Check ("trace mentions '" + $short + "'") ($t -match [regex]::Escape($short))
}
if ($hwnd -ne [IntPtr]::Zero) { [Win32GuiGpu]::Close($hwnd) }
if (-not $proc.WaitForExit(20000)) { $proc.Kill() } 

Write-Host "[3] ECM_GUI_NVML points nowhere: the GUI must degrade, not fail"
$env:ECM_GUI_NVML = 'Z:\definitely\not\nvml.dll'
Remove-Item $trace -ErrorAction SilentlyContinue
$proc2 = Start-Process -FilePath $Exe -ArgumentList @('-ini', $ini, '--trace') -PassThru
$hwnd2 = [IntPtr]::Zero
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline -and $hwnd2 -eq [IntPtr]::Zero) {
    Start-Sleep -Milliseconds 400
    $proc2.Refresh()
    if ($proc2.HasExited) { break }
    $hwnd2 = [Win32GuiGpu]::Find([uint32]$proc2.Id, 'ecm_gui')
}
Check "GUI still starts without NVML" ($hwnd2 -ne [IntPtr]::Zero)
Start-Sleep -Seconds 2
$t2 = Read-TextShared $trace
Check "trace reports NVML unavailable" ($t2 -match 'gpu: NVML unavailable')
Check "the reason is a readable message" ($t2 -match 'nvml\.dll not found|NVML')
if ($hwnd2 -ne [IntPtr]::Zero) { [Win32GuiGpu]::Close($hwnd2) }
if (-not $proc2.WaitForExit(20000)) { $proc2.Kill(); Check "GUI exits cleanly without NVML" $false "had to kill" }
else { Check "GUI exits cleanly without NVML" ($proc2.ExitCode -eq 0) ("exit code " + $proc2.ExitCode) }
Remove-Item Env:\ECM_GUI_NVML -ErrorAction SilentlyContinue

Write-Host ""
Write-Host ("passed: " + $script:pass + "   failed: " + $script:fail)
if ($script:fail -gt 0) { exit 1 }
exit 0
