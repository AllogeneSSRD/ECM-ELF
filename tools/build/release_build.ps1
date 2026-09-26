<#
.SYNOPSIS
    RELEASE build for ecm_cuda -- multi-arch build + one pruned exe per architecture, one command.

.DESCRIPTION
    Step 1 configures and builds `build_cuda_release` for every architecture you ship, with PTX
    embedded for ONE architecture only (the oldest you support) as the forward-compatibility
    fallback for GPUs newer than anything you compiled SASS for.

    Step 2 splits the result into one exe per architecture with tools\build\release_split.ps1.
    That split is necessary because nvprune cannot be applied to a linked executable:

        nvprune fatal : Input file '...ecm_cuda.exe' not relocatable.

    so the pipeline prunes the `.cu.obj` files and relinks (seconds per architecture).

    Keep ECM_CUDA_COMPRESS=OFF here: nvprune rewrites the embedded fatbins, and the verified
    split path uses uncompressed fatbins.  (The local test build is the opposite: compression ON,
    no PTX, one arch -- see tools\build\local_build.ps1.)

    Measured on a tier-restricted 6-arch build: multi-arch exe 3.3 MB; after the split the
    sm_75 binary (SASS + PTX) is 3.13 MB and every other architecture 978 KB.

.PARAMETER BuildDir   Multi-arch build directory (default build_cuda_release).
.PARAMETER OutDir     Where the per-arch exes land (default dist\cuda).
.PARAMETER Archs      Comma list of architectures to ship (default 75,86,89,90,100,120).
.PARAMETER PtxArch    Architecture that also keeps its PTX (default 75).
.PARAMETER Tiers      Restrict kernel tiers for a quick smoke test, e.g. "5120".
.PARAMETER Jobs       Concurrent nvcc processes (default 6).
.PARAMETER SkipBuild  Only run the split (the build directory must already be compiled).
.PARAMETER SkipSplit  Only build; do not produce the per-arch exes.

.NOTES
    A full 6-architecture build of the complete kernel set takes roughly 45 minutes on this
    machine (each architecture is ~450 s of nvcc work; more than ~6 concurrent nvcc does not
    help -- see docs/ECM_CGBN_OPTIMIZATION.md 8.8).
    CUDA 13.3 cannot target compute_60/61/70 at all (nvcc --list-gpu-arch stops at 75), so
    10-series Pascal needs a build produced with CUDA <= 12.x (merge it with fatbinary) or a
    separate legacy package.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\release_build.ps1
.EXAMPLE
    # quick end-to-end smoke test of the release pipeline (one tier, two architectures)
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\release_build.ps1 `
        -Tiers 5120 -Archs 75,89 -BuildDir build_reltest -OutDir .bench_tmp\reltest
#>
param(
    [string]$BuildDir = "build_cuda_release",
    [string]$OutDir = "dist\cuda",
    [string]$Archs = "75,86,89,90,100,120",
    [string]$PtxArch = "75",
    [string]$Tiers = "",
    [int]$Jobs = 6,
    [switch]$SkipBuild,
    [switch]$SkipSplit,
    [switch]$FullRebuild,
    [string]$Gmp = "",
    [string]$OpenSslRoot = "",
    [string]$VcVars = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not [System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
if (-not [System.IO.Path]::IsPathRooted($OutDir)) { $OutDir = Join-Path $repo $OutDir }
if (-not $Gmp) { $Gmp = Join-Path $repo "third_party\gmp-zen3\dist" }
if (-not $OpenSslRoot) { $OpenSslRoot = "D:/code/vcpkg/installed/x64-windows" }
if (-not $VcVars) {
    $cand = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending
    if (-not $cand) { throw "vcvars64.bat not found; pass -VcVars" }
    $VcVars = $cand[0].FullName
}

if (-not $SkipBuild) {
    $cfg = @(
        "-S", "`"$repo`"", "-B", "`"$BuildDir`"", '-G', '"NMake Makefiles"',
        "-DCMAKE_BUILD_TYPE=Release",
        "-DECM_ENABLE_CUDA=ON",
        "-DECM_CUDA_FULL_BUILD=ON",
        "-DECM_CUDA_ARCHITECTURES=`"$($Archs -replace ',', ';')`"",
        "-DECM_CUDA_PTX_ARCH=$PtxArch",
        "-DECM_CUDA_COMPRESS=OFF",
        "-DECM_TPB=128", "-DECM_MAX_ROTATION=1",
        "-DECM_MAXRREG_SMALL=0", "-DECM_MAXRREG_SUYAMA=0",
        "-DECM_REG_TARGET_FORCE=0", "-DECM_NO_PARAM2=0",
        "-DECM_TIERS=$Tiers",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DGMP_INCLUDE_DIR=$($Gmp -replace '\\','/')/include",
        "-DGMP_LIBRARY=$($Gmp -replace '\\','/')/lib/gmp.lib",
        "-DOPENSSL_ROOT_DIR=$OpenSslRoot"
    ) -join ' '

    Write-Host "== RELEASE rule: archs [$Archs], PTX for $PtxArch only, uncompressed (prunable) =="
    & cmd.exe /c "call `"$VcVars`" >nul 2>&1 && cmake $cfg" | Select-String -Pattern 'device code|PTX embedded|Error' |
        ForEach-Object { "   $($_.Line.Trim())" }
    if ($LASTEXITCODE -ne 0) { throw "configure failed" }

    if ($FullRebuild) {
        Get-ChildItem (Join-Path $BuildDir "CMakeFiles\ecm_cuda.dir\kernels\cuda") -Filter *.obj -ErrorAction SilentlyContinue |
            Remove-Item -Force
        Write-Host "   (CUDA objects removed)"
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "parallel_nvcc.ps1") `
        -BuildDir $BuildDir -Jobs $Jobs
}

if (-not $SkipSplit) {
    Write-Host "== splitting into one exe per architecture =="
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "release_split.ps1") `
        -BuildDir $BuildDir -OutDir $OutDir -Archs $Archs -PtxArch $PtxArch -VcVars $VcVars
    Write-Host "== done: see $OutDir =="
}
