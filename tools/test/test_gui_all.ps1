#Requires -Version 5.1
<#
.SYNOPSIS
    Runs the whole ecm_gui verification suite in one go (the same set the development
    rounds used) and prints one summary table.

.DESCRIPTION
    Two kinds of entry points are covered:

      * headless self-tests shipped in the binary: --selftest, --worker-selftest,
        --gpu-selftest, plus the two standalone unit-test executables;
      * scripted real-window tests in tools/test: smoke (window lifecycle, layout, font,
        CJK), workers (fake worker supervision, table geometry, graceful stop), gpu (NVML),
        gpu_curves (real worker load), real_workers (real ecm_cuda + two cards), results
        (results.json.txt / results.txt), exit_checkpoint (checkpoint before exit), plus the
        driver-side tests (hit fields, worker/worktodo sections).

    Each test prints "passed: N   failed: M" and exits non-zero on failure; this script
    aggregates them, keeps every log under tools/test/_run/suite_<timestamp>/, and exits
    non-zero if anything failed.

.PARAMETER SkipGpu
    Skip the tests that need an NVIDIA GPU / NVML (gpu, gpu_curves, real_workers, results,
    hit_fields, and --gpu-selftest).

.PARAMETER Only
    Run only tests whose name matches this wildcard (e.g. -Only '*smoke*').

.PARAMETER List
    Print the test names and exit.

.PARAMETER TimeoutSeconds
    Per-test timeout (default 900 s). A test that overruns is killed and counted as failed.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_gui_all.ps1
.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_gui_all.ps1 -SkipGpu
#>
param(
    [string]$Build = "",
    [switch]$SkipGpu,
    [string]$Only = "*",
    [switch]$List,
    [int]$TimeoutSeconds = 900,
    [switch]$KeepGoing
)

$ErrorActionPreference = 'Continue'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# Where the binaries are: build_gui first (the configuration the docs use), then the
# Visual Studio / CUDA builds.
if (-not $Build) {
    foreach ($cand in @("$repoRoot\build_gui", "$repoRoot\build_vs18\Release", "$repoRoot\build_cuda_cmake")) {
        if (Test-Path (Join-Path $cand 'ecm_gui.exe')) { $Build = $cand; break }
    }
}
if (-not $Build -or -not (Test-Path (Join-Path $Build 'ecm_gui.exe'))) {
    Write-Host "FAIL: ecm_gui.exe not found. Build it first:" -ForegroundColor Red
    Write-Host "  cmake --build build_gui --target ecm_gui ecm_gui_fake_worker ecm_gui_log_parse_test ecm_gui_results_test"
    exit 2
}
$exe = Join-Path $Build 'ecm_gui.exe'
$fake = Join-Path $Build 'ecm_gui_fake_worker.exe'

# Real ecm_cuda.exe for the GPU tests: the CUDA build first, then a release layout.
$ecmCuda = ""
foreach ($cand in @("$repoRoot\build_cuda_cmake\ecm_cuda.exe", "$repoRoot\build_vs18\Release\ecm_cuda.exe")) {
    if (Test-Path $cand) { $ecmCuda = $cand; break }
}

# ---------------------------------------------------------------- test definitions ----
# kind: "exe"    -> run the binary with these arguments
#       "script" -> run the PowerShell script
# exe:  which executable the script wants: "gui" = -Exe <ecm_gui.exe>,
#       "driver" = -Exe <ecm_cuda.exe> (driver-side scripts), "" = no -Exe
#       NOTE: passing the wrong one is not harmless -- test_worker_sections.ps1 takes the
#       DRIVER in -Exe, so handing it the GUI made it run the GUI as a "worker" and left a
#       window sitting on the desktop (measured 2026-09-29, the user closed it by hand).
# cuda: add -EcmCuda <ecm_cuda.exe> (only for scripts that declare it)
$tests = @(
    @{ name = 'selftest';        kind = 'exe';    target = $exe;     args = @('--selftest');        gpu = $false; what = 'ini / localization / fonts / state keys (headless)' }
    @{ name = 'worker-selftest'; kind = 'exe';    target = $exe;     args = @('--worker-selftest'); gpu = $false; what = 'worker supervision against the fake worker (headless)' }
    @{ name = 'gpu-selftest';    kind = 'exe';    target = $exe;     args = @('--gpu-selftest');    gpu = $true;  what = 'NVML fields, cross-checked against nvidia-smi' }
    @{ name = 'log-parse';       kind = 'exe';    target = (Join-Path $Build 'ecm_gui_log_parse_test.exe');   args = @(); gpu = $false; what = 'progress/event/hit line parsing (unit)' }
    @{ name = 'results-unit';    kind = 'exe';    target = (Join-Path $Build 'ecm_gui_results_test.exe');     args = @(); gpu = $false; what = 'results.json.txt + results.txt (unit)' }
    @{ name = 'smoke';           kind = 'script'; target = 'test_gui_smoke.ps1';          exe = 'gui';    gpu = $false; what = 'window lifecycle, minimize/restore, layout, font, CJK' }
    @{ name = 'workers';         kind = 'script'; target = 'test_gui_workers.ps1';        exe = 'gui';    fake = $true; gpu = $false; what = 'fake-worker supervision, table geometry, graceful stop' }
    @{ name = 'exit-checkpoint'; kind = 'script'; target = 'test_gui_exit_checkpoint.ps1'; exe = 'gui'; cuda = $true; gpu = $true; what = 'exit confirmation + checkpoint written before exit (real driver)' }
    @{ name = 'cjk-pixels';      kind = 'script'; target = 'test_gui_cjk_pixels.ps1';     exe = 'gui';    gpu = $false; what = 'Chinese rendering verified in the window pixels' }
    @{ name = 'gpu-panel';       kind = 'script'; target = 'test_gui_gpu.ps1';            exe = 'gui';    gpu = $true;  what = 'GPU panel + NVML degradation' }
    @{ name = 'gpu-curves';      kind = 'script'; target = 'test_gui_gpu_curves.ps1';     exe = 'gui';    cuda = $true; gpu = $true;  what = 'power/clock curves really move under load' }
    @{ name = 'real-workers';    kind = 'script'; target = 'test_gui_real_workers.ps1';   exe = 'gui';    cuda = $true; gpu = $true;  what = 'real ecm_cuda on two cards, worktodo sections' }
    @{ name = 'results-e2e';     kind = 'script'; target = 'test_gui_results.ps1';        exe = 'gui';    cuda = $true; gpu = $true;  what = 'results dual file, two real runs' }
    @{ name = 'hit-fields';      kind = 'script'; target = 'test_hit_fields.ps1';         exe = '';       cuda = $true; gpu = $true;  what = 'D3 hit line fields (real driver, queue mode)' }
    @{ name = 'worker-sections'; kind = 'script'; target = 'test_worker_sections.ps1';    exe = 'driver'; gpu = $true;  what = 'D1/D2 ini + worktodo [Worker #N] sections' }
)

if ($List) {
    foreach ($t in $tests) { "{0,-16} {1}{2}" -f $t.name, $t.what, $(if ($t.gpu) { '  [GPU]' } else { '' }) }
    exit 0
}

$selected = @($tests | Where-Object { $_.name -like $Only -and (-not $SkipGpu -or -not $_.gpu) })
if ($selected.Count -eq 0) { Write-Host "no test matches '$Only'" -ForegroundColor Red; exit 2 }

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logDir = Join-Path $PSScriptRoot "_run\suite_$stamp"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# ---------------------------------------------------------------- stray-process guard --
# A test that hangs must not leave a window on the user's desktop: before and after the
# run (and after a failure) kill any ecm_gui / fake worker / ecm_cuda whose executable
# lives under tools/test/_run. Only sandbox copies are touched -- never a real build or a
# production directory.
function Remove-StrayTestProcesses([string]$reason) {
    $killed = @()
    foreach ($name in @('ecm_gui', 'ecm_gui_fake_worker', 'ecm_cuda')) {
        foreach ($p in @(Get-Process -Name $name -ErrorAction SilentlyContinue)) {
            $path = ""
            try { $path = $p.MainModule.FileName } catch { $path = "" }
            if ($path -like "*\tools\test\_run\*") {
                $killed += ("{0}({1})" -f $name, $p.Id)
                try { $p.Kill() } catch { }
            }
        }
    }
    if ($killed.Count -gt 0) {
        Write-Host ("  cleanup (" + $reason + "): killed " + ($killed -join ', ')) -ForegroundColor Yellow
    }
    return $killed.Count
}
[void](Remove-StrayTestProcesses 'before the suite')

Write-Host "ecm_gui test suite"
Write-Host ("  build  : " + $Build)
Write-Host ("  ecm_cuda: " + $(if ($ecmCuda) { $ecmCuda } else { '<not found - GPU tests will fail>' }))
Write-Host ("  logs   : " + $logDir)
Write-Host ("  tests  : " + $selected.Count + $(if ($SkipGpu) { " (GPU tests skipped)" } else { "" }))
Write-Host ""

$results = New-Object System.Collections.ArrayList
$index = 0
foreach ($t in $selected) {
    $index++
    $logPath = Join-Path $logDir ($t.name + '.log')
    Write-Host ("[{0}/{1}] {2} -- {3}" -f $index, $selected.Count, $t.name, $t.what)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $exit = -1
    $timedOut = $false
    # Runner notes (both measured on this host, 2026-09-29):
    #   * Start-Process -PassThru does NOT give a usable ExitCode here (always empty), and
    #     it rejects an empty -ArgumentList, so tests are invoked with '&' + $LASTEXITCODE;
    #   * Start-Job cannot relay $LASTEXITCODE out of the job either, so the timeout is not
    #     implemented by killing the test: a watchdog job kills only the SANDBOX processes
    #     (ecm_gui / fake worker / ecm_cuda under tools/test/_run) when the deadline passes.
    #     Every test has its own internal deadlines, so this is a safety net for windows
    #     left on the desktop, not the primary mechanism.
    $watchdog = Start-Job -ArgumentList $TimeoutSeconds -ScriptBlock {
        param($seconds)
        Start-Sleep -Seconds $seconds
        $killed = 0
        foreach ($name in @('ecm_gui', 'ecm_gui_fake_worker', 'ecm_cuda')) {
            foreach ($p in @(Get-Process -Name $name -ErrorAction SilentlyContinue)) {
                $path = ""
                try { $path = $p.MainModule.FileName } catch { }
                if ($path -like "*\tools\test\_run\*") {
                    try { $p.Kill(); $killed++ } catch { }
                }
            }
        }
        "fired:$killed"
    }
    try {
        if ($t.kind -eq 'exe') {
            if (-not (Test-Path $t.target)) {
                "MISSING: $($t.target)" | Set-Content $logPath
                $exit = 2
            } else {
                $targs = @($t.args)
                if ($targs.Count -gt 0) {
                    & $t.target @targs *>&1 | Tee-Object -FilePath $logPath | Out-Null
                } else {
                    & $t.target *>&1 | Tee-Object -FilePath $logPath | Out-Null
                }
                $exit = $LASTEXITCODE
            }
        } else {
            $scriptPath = Join-Path $PSScriptRoot $t.target
            if (-not (Test-Path $scriptPath)) {
                "MISSING: $scriptPath" | Set-Content $logPath
                $exit = 2
            } else {
                $argv = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $scriptPath)
                switch ($t.exe) {
                    'gui'    { $argv += @('-Exe', $exe) }
                    'driver' { if ($ecmCuda) { $argv += @('-Exe', $ecmCuda) } }
                    default  { }
                }
                if ($t.fake -and (Test-Path $fake)) { $argv += @('-Fake', $fake) }
                if ($t.cuda -and $ecmCuda) { $argv += @('-EcmCuda', $ecmCuda) }
                & powershell @argv *>&1 | Tee-Object -FilePath $logPath | Out-Null
                $exit = $LASTEXITCODE
            }
        }
    } catch {
        $_ | Out-String | Add-Content $logPath
        $exit = 3
    }
    # Did the watchdog fire? (It only fires if the test overran its budget.)
    $fired = @(Receive-Job $watchdog -ErrorAction SilentlyContinue)
    if ($fired.Count -gt 0) {
        $timedOut = $true
        Add-Content $logPath ("TIMEOUT: the watchdog fired after " + $TimeoutSeconds + " s (" + $fired[-1] + ")")
    }
    Stop-Job $watchdog -ErrorAction SilentlyContinue
    Remove-Job $watchdog -Force -ErrorAction SilentlyContinue
    $sw.Stop()
    [void](Remove-StrayTestProcesses ("after " + $t.name))

    $text = if (Test-Path $logPath) { Get-Content $logPath -Raw } else { '' }
    $m = [regex]::Match($text, 'passed:\s*(\d+)\s+failed:\s*(\d+)')
    $passed = if ($m.Success) { [int]$m.Groups[1].Value } else { 0 }
    $failed = if ($m.Success) { [int]$m.Groups[2].Value } else { 0 }
    $ok = ($exit -eq 0) -and ($m.Success) -and ($failed -eq 0) -and (-not $timedOut)
    [void]$results.Add([pscustomobject]@{
            Name    = $t.name
            Ok      = $ok
            Passed  = $passed
            Failed  = $failed
            Seconds = [Math]::Round($sw.Elapsed.TotalSeconds, 1)
            Exit    = $exit
            Log     = $logPath
        })
    if ($ok) {
        Write-Host ("        ok    passed={0} ({1}s)" -f $passed, [Math]::Round($sw.Elapsed.TotalSeconds, 1)) -ForegroundColor Green
    } else {
        Write-Host ("        FAIL  exit={0} passed={1} failed={2}  log: {3}" -f $exit, $passed, $failed, $logPath) -ForegroundColor Red
        if (Test-Path $logPath) {
            Select-String -Path $logPath -Pattern 'FAIL' | Select-Object -First 6 | ForEach-Object {
                Write-Host ("              " + $_.Line.Trim()) -ForegroundColor DarkYellow
            }
        }
        if (-not $KeepGoing) {
            Write-Host ""
            Write-Host "stopped at the first failure (-KeepGoing to run the rest)" -ForegroundColor Yellow
            break
        }
    }
}

Write-Host ""
Write-Host "================ summary ================"
$results | Format-Table Name, Ok, Passed, Failed, Seconds, Exit -AutoSize
[void](Remove-StrayTestProcesses 'after the suite')
$totalPassed = ($results | Measure-Object -Property Passed -Sum).Sum
$bad = @($results | Where-Object { -not $_.Ok })
$notRun = $selected.Count - $results.Count
Write-Host ("tests run   : {0} of {1}" -f $results.Count, $selected.Count)
Write-Host ("checks      : {0} passed, {1} failed" -f $totalPassed, ($results | Measure-Object -Property Failed -Sum).Sum)
Write-Host ("tests failed: {0}" -f $bad.Count)
if ($notRun -gt 0) { Write-Host ("not run     : {0}" -f $notRun) }
Write-Host ("logs        : " + $logDir)
if ($bad.Count -gt 0) { exit 1 }
exit 0
