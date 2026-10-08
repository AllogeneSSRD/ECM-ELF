#Requires -Version 5.1
<#
.SYNOPSIS
    Builds ecm_gui (and the GUI's test helpers) with one command, from a plain PowerShell.

.DESCRIPTION
    Why this exists: the GUI is built with CMake's "NMake Makefiles" generator, and
    `cmake --build build_gui --target ecm_gui` from an ordinary PowerShell fails with

        no such file or directory
        CMake Error: Generator: build tool execution failed, command was: nmake -f Makefile /nologo ecm_gui

    because `nmake` only exists inside the Visual Studio developer environment. This script

      1. finds vcvars64.bat (Visual Studio 18/17/16, Community or BuildTools),
      2. configures <build dir> when needed (or when -Reconfigure is given),
      3. builds the requested targets inside a shell that has that environment,
      4. reports the produced exe and optionally runs --selftest.

    Everything the GUI needs is in the repository or already on this machine; the paths are
    detected, and can be overridden (see the parameters).

.PARAMETER BuildDir
    Build directory (default: build_gui). Created/configured on the first run.

.PARAMETER Targets
    CMake targets to build. Default: the GUI plus the two unit-test binaries and the fake
    worker (they are what tools\test scripts need).

.PARAMETER Reconfigure
    Delete CMakeCache.txt and configure again (use after changing -Gmp / -OpenSslRoot /
    -Generator or when CMake complains about a stale cache).

.PARAMETER Clean
    Remove the whole build directory first (a true from-scratch build).

.PARAMETER Generator
    CMake generator override, e.g. "Visual Studio 17 2022" for an IDE build. Default:
    "NMake Makefiles" (the configuration the documentation and tests use).

.PARAMETER Gmp / OpenSslRoot / ImGuiDir
    Dependency overrides. Defaults: third_party\gmp-zen3\dist, D:/code/vcpkg/installed/x64-windows,
    third_party/imgui.

.PARAMETER Selftest
    Run `ecm_gui.exe --selftest` when the build succeeded (fast sanity check, no window).

.PARAMETER Smoke
    Also run tools\test\test_gui_smoke.ps1 (a real window appears for a few seconds).

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\dev\build_gui.ps1
.EXAMPLE
    # clean rebuild plus the self-test
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\dev\build_gui.ps1 -Clean -Selftest
.EXAMPLE
    # an IDE-friendly project instead
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\dev\build_gui.ps1 -Generator "Visual Studio 17 2022"
#>
param(
    [string]$BuildDir = "build_gui",
    [string[]]$Targets = @('ecm_gui', 'ecm_gui_fake_worker', 'ecm_gui_log_parse_test', 'ecm_gui_results_test'),
    [switch]$Reconfigure,
    [switch]$Clean,
    [string]$Generator = "NMake Makefiles",
    [string]$Gmp = "",
    [string]$OpenSslRoot = "",
    [string]$ImGuiDir = "",
    [string]$VcVars = "",
    [switch]$Selftest,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if (-not [System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }

# ------------------------------------------------------------------ dependencies --------
if (-not $Gmp) { $Gmp = Join-Path $repo "third_party\gmp-zen3\dist" }
if (-not $OpenSslRoot) {
    foreach ($cand in @("D:/code/vcpkg/installed/x64-windows",
                        "C:/code/vcpkg/installed/x64-windows",
                        (Join-Path $repo "third_party\vcpkg\installed\x64-windows"))) {
        if (Test-Path (Join-Path $cand 'include\openssl\ssl.h')) { $OpenSslRoot = $cand; break }
    }
}
if (-not $ImGuiDir) { $ImGuiDir = Join-Path $repo "third_party\imgui" }

$missing = @()
if (-not (Test-Path (Join-Path $Gmp 'include\gmp.h'))) { $missing += "GMP headers in $Gmp" }
if (-not (Test-Path (Join-Path $Gmp 'lib\gmp.lib'))) { $missing += "GMP import library in $Gmp\lib" }
if (-not $OpenSslRoot -or -not (Test-Path (Join-Path $OpenSslRoot 'include\openssl\ssl.h'))) {
    $missing += "OpenSSL in '$OpenSslRoot' (pass -OpenSslRoot <dir>)"
}
if (-not (Test-Path (Join-Path $ImGuiDir 'imgui.h'))) {
    $missing += "vendored Dear ImGui in $ImGuiDir (see third_party/imgui/README.md)"
}
if ($missing.Count -gt 0) {
    Write-Host "cannot configure: missing dependencies" -ForegroundColor Red
    foreach ($m in $missing) { Write-Host ("  - " + $m) }
    exit 2
}

# ------------------------------------------------------------------ developer env -------
if (-not $VcVars) {
    $cand = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending
    if (-not $cand) {
        Write-Host "vcvars64.bat not found (Visual Studio C++ build tools)." -ForegroundColor Red
        Write-Host "Install 'Desktop development with C++' (VS 2022/2026 Community or Build Tools),"
        Write-Host "or pass -VcVars <path\to\vcvars64.bat>."
        exit 2
    }
    # Prefer a full Visual Studio over Build Tools (both work; the full one also has the IDE).
    $preferred = $cand | Where-Object { $_.FullName -notmatch 'BuildTools' } | Select-Object -First 1
    $VcVars = if ($preferred) { $preferred.FullName } else { $cand[0].FullName }
}
Write-Host "== ecm_gui build =="
Write-Host ("   repo      : " + $repo)
Write-Host ("   build dir : " + $BuildDir)
Write-Host ("   generator : " + $Generator)
Write-Host ("   vcvars    : " + $VcVars)
Write-Host ("   gmp       : " + $Gmp)
Write-Host ("   openssl   : " + $OpenSslRoot)
Write-Host ("   imgui     : " + $ImGuiDir)

if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "   (removing the build directory: full rebuild)"
    Remove-Item $BuildDir -Recurse -Force
}
if ($Reconfigure) {
    foreach ($f in @('CMakeCache.txt')) {
        $p = Join-Path $BuildDir $f
        if (Test-Path $p) { Remove-Item $p -Force; Write-Host ("   (removed " + $f + ")") }
    }
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

# Runs a command line through cmd.exe with the developer environment loaded and returns
# @{ exit; text }. Native stderr must not become a PowerShell terminating error (with
# $ErrorActionPreference = Stop, cmake's stderr aborted the script before it could report
# the real failure), hence the local Continue + explicit capture.
function Invoke-DevCmd([string]$commandLine) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $text = (& cmd.exe /c "call `"$VcVars`" >nul 2>&1 && $commandLine" 2>&1 | Out-String)
        return @{ exit = $LASTEXITCODE; text = $text }
    } finally {
        $ErrorActionPreference = $prev
    }
}

# ------------------------------------------------------------------ configure -----------
$needConfig = -not (Test-Path (Join-Path $BuildDir 'CMakeCache.txt'))
if ($needConfig) {
    Write-Host "== configure =="
    $defs = @(
        "-S", "`"$repo`"", "-B", "`"$BuildDir`"", "-G", "`"$Generator`"",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DECM_BUILD_GUI=ON",
        "-DECM_BUILD_TOOLS=OFF",              # the GUI does not need the bench/test tools
        "-DECM_IMGUI_DIR=`"$($ImGuiDir -replace '\\','/')`"",
        "-DGMP_INCLUDE_DIR=`"$($Gmp -replace '\\','/')/include`"",
        "-DGMP_LIBRARY=`"$($Gmp -replace '\\','/')/lib/gmp.lib`"",
        "-DOPENSSL_ROOT_DIR=`"$($OpenSslRoot -replace '\\','/')`""
    ) -join ' '
    $r = Invoke-DevCmd "cmake $defs"
    ($r.text -split "`r?`n") | Select-String -Pattern 'gui:|Build files|Error|error|warning: ' |
        ForEach-Object { "   " + $_.Line.Trim() }
    if ($r.exit -ne 0) {
        Write-Host "configure failed (exit $($r.exit)) -- full output:" -ForegroundColor Red
        ($r.text -split "`r?`n") | ForEach-Object { Write-Host ("   " + $_) }
        exit 1
    }
    # The root CMakeLists prints "-- gui: ecm_gui enabled (...) / disabled / skipped".
    if ($r.text -notmatch 'gui:.*enabled') {
        Write-Host "the GUI target was not enabled (see 'gui:' above):" -ForegroundColor Red
        Write-Host "  * 'gui: skipped'  -> third_party/imgui/imgui.h is missing"
        Write-Host "  * 'gui: disabled' -> -DECM_BUILD_GUI=OFF"
        exit 1
    }
} else {
    Write-Host "== configure skipped (CMakeCache.txt exists; use -Reconfigure to redo it) =="
}

# ------------------------------------------------------------------ build ---------------
Write-Host "== build =="
$targetArgs = ($Targets | ForEach-Object { "--target $_" }) -join ' '
$r = Invoke-DevCmd "cmake --build `"$BuildDir`" $targetArgs"
($r.text -split "`r?`n") | Select-String -Pattern 'error|Error|warning C|Built target|Linking' |
    ForEach-Object { "   " + $_.Line.Trim() }
if ($r.exit -ne 0) {
    Write-Host "build failed (exit $($r.exit)) -- full output:" -ForegroundColor Red
    ($r.text -split "`r?`n") | ForEach-Object { Write-Host ("   " + $_) }
    exit 1
}

$exe = Join-Path $BuildDir 'ecm_gui.exe'
if (-not (Test-Path $exe)) {
    Write-Host "build reported success but $exe is missing" -ForegroundColor Red
    exit 1
}
"   ecm_gui.exe: {0:N0} B, {1}" -f (Get-Item $exe).Length, (Get-Item $exe).LastWriteTime

# ------------------------------------------------------------------ verify --------------
if ($Selftest) {
    Write-Host "== selftest =="
    & $exe --selftest | Select-String -Pattern '^passed:|FAIL' | ForEach-Object { "   " + $_.Line.Trim() }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "selftest failed" -ForegroundColor Red
        exit 1
    }
}
if ($Smoke) {
    Write-Host "== smoke test (a window will appear for a few seconds) =="
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo 'tools\test\test_gui_smoke.ps1') -Exe $exe |
        Select-String -Pattern '^passed:|FAIL' | ForEach-Object { "   " + $_.Line.Trim() }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "smoke test failed" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host ("done: " + $exe) -ForegroundColor Green
Write-Host "next: run it next to your ecm.ini (and ecm_cuda.exe), or run the whole test suite:"
Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_gui_all.ps1"
exit 0
