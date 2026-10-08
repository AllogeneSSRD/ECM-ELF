#Requires -Version 5.1
<#
.SYNOPSIS
    Builds arbitrary CMake targets of this repository from a plain PowerShell.

.DESCRIPTION
    The repository's build directories use CMake's "NMake Makefiles" generator, so
    `cmake --build <dir> --target <t>` only works inside the Visual Studio developer
    environment (otherwise: "Generator: build tool execution failed, command was:
    nmake -f Makefile"). tools\build\dev\build_gui.ps1 wraps that for the GUI; this script
    does the same for any other build directory, which is what the CUDA driver
    (build_cuda_cmake -> ecm_cuda) needs.

    It
      1. finds vcvars64.bat (Visual Studio 18/17/16, Community or BuildTools),
      2. configures the build directory when it has no CMakeCache.txt yet (pass
         -ConfigureArgs for the cache entries that build needs),
      3. builds the requested targets in a shell that has the developer environment,
      4. reports every produced .exe with size and timestamp, and fails loudly on error.

.PARAMETER BuildDir
    Build directory, absolute or relative to the repository root. It is created when
    missing. Required.

.PARAMETER Targets
    CMake targets to build. Default: ecm_cuda.

.PARAMETER ConfigureArgs
    Extra `-D...`/`-G...` arguments used only when the directory still has to be
    configured, e.g. -ConfigureArgs '-DECM_BUILD_TOOLS=OFF'.

.PARAMETER Filter
    Regular expression applied to the build output before it is echoed (default:
    errors, warnings and linked targets). Use '.' for everything.

.PARAMETER VcVars
    Explicit path to vcvars64.bat when the automatic search fails.

.PARAMETER TimeoutMinutes
    Abort a build that runs longer than this (default 60). The CUDA targets compile a
    lot of cubins; raise it for a from-scratch full build.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\dev\build_dev.ps1 -BuildDir build_cuda_cmake
.EXAMPLE
    # what the GUI needs, when build_gui.ps1 is not the right entry point
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\dev\build_dev.ps1 `
        -BuildDir build_gui -Targets ecm_gui,ecm_gui_fake_worker
#>
param(
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [string[]]$Targets = @('ecm_cuda'),
    [string]$ConfigureArgs = '',
    [switch]$Reconfigure,
    [string]$Filter = 'error|Error|warning C|Built target|Linking|nvcc fatal',
    [string]$VcVars = '',
    [int]$TimeoutMinutes = 60
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if (-not [System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }

if (-not $VcVars) {
    $cand = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending
    if (-not $cand) {
        Write-Host "vcvars64.bat not found (Visual Studio C++ build tools)." -ForegroundColor Red
        Write-Host "Install 'Desktop development with C++' or pass -VcVars <path\to\vcvars64.bat>."
        exit 2
    }
    $preferred = $cand | Where-Object { $_.FullName -notmatch 'BuildTools' } | Select-Object -First 1
    $VcVars = if ($preferred) { $preferred.FullName } else { $cand[0].FullName }
}

Write-Host "== build =="
Write-Host ("   repo      : " + $repo)
Write-Host ("   build dir : " + $BuildDir)
Write-Host ("   targets   : " + ($Targets -join ', '))
Write-Host ("   vcvars    : " + $VcVars)

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

# Native stderr must not become a terminating error under $ErrorActionPreference = Stop.
function Invoke-DevCmd([string]$commandLine, [int]$timeoutMin) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $job = Start-Job -ScriptBlock {
            param($cl, $dir)
            Set-Location $dir
            & cmd.exe /c $cl 2>&1 | Out-String
        } -ArgumentList $commandLine, $repo
        if (-not (Wait-Job $job -Timeout ($timeoutMin * 60))) {
            Stop-Job $job -ErrorAction SilentlyContinue
            Remove-Job $job -Force -ErrorAction SilentlyContinue
            return @{ exit = 124; text = "timed out after $timeoutMin minutes: $commandLine" }
        }
        $text = Receive-Job $job
        Remove-Job $job -Force -ErrorAction SilentlyContinue
        return @{ exit = 0; text = $text }
    } finally {
        $ErrorActionPreference = $prev
    }
}

# The caller must be able to see a non-zero exit code, and Start-Job cannot relay it,
# so the command line ends with an explicit marker line that we parse here.
function Invoke-DevCmdChecked([string]$commandLine) {
    $wrapped = "call `"$VcVars`" >nul 2>&1 && ($commandLine) & echo ECM_DEV_EXIT=%ERRORLEVEL%"
    $r = Invoke-DevCmd $wrapped $TimeoutMinutes
    $m = [regex]::Matches($r.text, 'ECM_DEV_EXIT=(\d+)')
    if ($m.Count -gt 0) { $r.exit = [int]$m[$m.Count - 1].Groups[1].Value }
    $r.text = $r.text -replace 'ECM_DEV_EXIT=\d+', ''
    return $r
}

$needConfig = -not (Test-Path (Join-Path $BuildDir 'CMakeCache.txt'))
if ($Reconfigure) {
    $cache = Join-Path $BuildDir 'CMakeCache.txt'
    if (Test-Path $cache) { Remove-Item $cache -Force; Write-Host "   (removed CMakeCache.txt)" }
    $needConfig = $true
}
if ($needConfig) {
    Write-Host "== configure =="
    if (-not $ConfigureArgs) {
        Write-Host "   (no -ConfigureArgs given: CMake will use its defaults)" -ForegroundColor Yellow
    }
    $r = Invoke-DevCmdChecked "cmake -S `"$repo`" -B `"$BuildDir`" -G `"NMake Makefiles`" -DCMAKE_BUILD_TYPE=Release $ConfigureArgs"
    ($r.text -split "`r?`n") | Select-String -Pattern 'Build files|Error|error' | ForEach-Object { "   " + $_.Line.Trim() }
    if ($r.exit -ne 0) {
        Write-Host "configure failed (exit $($r.exit)) -- full output:" -ForegroundColor Red
        ($r.text -split "`r?`n") | ForEach-Object { Write-Host ("   " + $_) }
        exit 1
    }
} else {
    Write-Host "== configure skipped (CMakeCache.txt exists) =="
}

$targetArgs = ($Targets | ForEach-Object { "--target $_" }) -join ' '
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$r = Invoke-DevCmdChecked "cmake --build `"$BuildDir`" $targetArgs"
$sw.Stop()
($r.text -split "`r?`n") | Select-String -Pattern $Filter | ForEach-Object { "   " + $_.Line.Trim() }
Write-Host ("   elapsed: {0:N1} s" -f $sw.Elapsed.TotalSeconds)
if ($r.exit -ne 0) {
    Write-Host "build failed (exit $($r.exit)) -- full output:" -ForegroundColor Red
    ($r.text -split "`r?`n") | ForEach-Object { Write-Host ("   " + $_) }
    exit 1
}

Write-Host "== produced =="
$any = $false
foreach ($t in $Targets) {
    $exe = Join-Path $BuildDir ($t + '.exe')
    if (Test-Path $exe) {
        $f = Get-Item $exe
        "   {0}: {1:N0} B, {2}" -f $f.Name, $f.Length, $f.LastWriteTime
        $any = $true
    }
}
if (-not $any) {
    Write-Host "   (no .exe matched the target names; check them)" -ForegroundColor Yellow
}
exit 0
