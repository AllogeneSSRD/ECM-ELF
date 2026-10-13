# =============================================================================
# build_and_run.ps1 - Configure + build (Release, vcpkg) + run an ECM stage-1 test.
#
# Examples:
#   .\build_and_run.ps1
#   .\build_and_run.ps1 -Reconfigure
#   .\build_and_run.ps1 -SkipBuild -Curves 256 -B1 1e5
#   .\build_and_run.ps1 -ExtraArgs "--special-mult generic"
#   .\build_and_run.ps1 -ShowKernel        # build, then ecm --showkernel
#
# Or via the .bat launcher (handles ExecutionPolicy):  build_and_run.bat -Reconfigure
# =============================================================================
param(
    [string]$BuildDir    = "build_rel",
    [string]$BuildConfig = "Release",
    [string]$Target      = "ecm",
    [string]$Toolchain   = "D:/code/vcpkg/scripts/buildsystems/vcpkg.cmake",
    [string]$OpenSSLRoot = "D:/code/vcpkg/installed/x64-windows",
    [switch]$Reconfigure,                 # force re-run of cmake configure
    [switch]$SkipBuild,                   # skip configure+build, just run
    [switch]$SkipRun,                     # configure+build only
    [switch]$ShowKernel,                  # run 'ecm --showkernel' instead of a factor run
    # --- run parameters (mirror the documented smoke test) ---
    [string]$N_expr  = "2^421-1",
    [int]   $Device  = 1,
    [string]$Sigma   = "1707370477",
    [int]   $Curves  = 64,
    [string]$B1      = "1e4",
    [string]$ExtraArgs = ""               # appended verbatim, e.g. "--special-mult generic"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$buildPath = Join-Path $root $BuildDir
Write-Host "============================================================"
Write-Host " ECM build + run"
Write-Host " Root       = $root"
Write-Host " Build dir  = $buildPath  ($BuildConfig)"
Write-Host "============================================================"

# --- locate cmake (PATH first, then the vcpkg-downloaded copy) ---
$cmake = $null
if (Get-Command cmake -ErrorAction SilentlyContinue) {
    $cmake = "cmake"
} else {
    $fallback = "D:\code\vcpkg\downloads\tools\cmake-4.3.2-windows\cmake-4.3.2-windows-x86_64\bin\cmake.exe"
    if (Test-Path $fallback) { $cmake = $fallback }
}
if (-not $cmake) {
    Write-Host "[ERROR] cmake not found on PATH and vcpkg fallback missing." -ForegroundColor Red
    exit 1
}
Write-Host " cmake      = $cmake"

if (-not $SkipBuild) {
    # --- configure (only if needed) ---
    $cacheFile = Join-Path $buildPath "CMakeCache.txt"
    if ($Reconfigure -or -not (Test-Path $cacheFile)) {
        Write-Host "`n== Configure ==" -ForegroundColor Cyan
        $cfgArgs = @(
            '-S', $root,
            '-B', $buildPath,
            "-DCMAKE_BUILD_TYPE=$BuildConfig",
            "-DCMAKE_TOOLCHAIN_FILE=$Toolchain",
            "-DOPENSSL_ROOT_DIR=$OpenSSLRoot"
        )
        & $cmake @cfgArgs
        if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] configure failed ($LASTEXITCODE)" -ForegroundColor Red; exit $LASTEXITCODE }
    } else {
        Write-Host "`n== Configure skipped (cache exists; pass -Reconfigure to redo) =="
    }

    # --- build ---
    Write-Host "`n== Build (target: $Target) ==" -ForegroundColor Cyan
    & $cmake --build $buildPath --config $BuildConfig --target $Target
    if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] build failed ($LASTEXITCODE)" -ForegroundColor Red; exit $LASTEXITCODE }
}

# --- locate the produced exe ---
$exe = Join-Path $buildPath "$BuildConfig\ecm.exe"
if (-not (Test-Path $exe)) {
    Write-Host "[ERROR] ecm.exe not found: $exe" -ForegroundColor Red
    Write-Host "        (build may have failed, or it landed elsewhere)" -ForegroundColor Red
    exit 1
}
Write-Host "`n== Built: $exe ==" -ForegroundColor Green

if ($SkipRun) { exit 0 }

# --- run ---
if ($ShowKernel) {
    Write-Host "`n== Run: ecm.exe --showkernel ==" -ForegroundColor Cyan
    & $exe --showkernel
    Write-Host "`n== exit code: $LASTEXITCODE =="
    exit $LASTEXITCODE
}

$ecmArgs = @('-v','-d',"$Device",'--curve','mont','--target','gpu','-sigma',"3:$Sigma",'-gpucurves',"$Curves","$B1",'0')
if ($ExtraArgs.Trim().Length -gt 0) { $ecmArgs += ($ExtraArgs -split '\s+') }

Write-Host "`n== Run ==" -ForegroundColor Cyan
Write-Host ("  echo ""({0})"" | ecm.exe {1}" -f $N_expr, ($ecmArgs -join ' '))
"($N_expr)" | & $exe @ecmArgs
$code = $LASTEXITCODE
Write-Host "`n== exit code: $code =="
exit $code
