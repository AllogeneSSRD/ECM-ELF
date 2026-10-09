<#
.SYNOPSIS
    Local Stage1 build for ecm_cuda, using the general Montgomery baseline.

.DESCRIPTION
    Configures and builds `build_cuda_cmake`; defaults are described in docs/usage/BUILD.md:

        compression ON      -Xfatbin -compress-all
        no PTX              ECM_CUDA_EMBED_PTX=OFF        (needs exactly sm_<Arch>)
        one architecture    sm_89                         (override with -Arch)
        full kernel set, all tiers, param2 enabled

    The compile step uses tools\build\internal\parallel_nvcc.ps1 for independent CUDA TUs.

.PARAMETER BuildDir   Build directory (default build_cuda_cmake, the local test dir).
.PARAMETER Arch       CUDA architecture (default 89).
.PARAMETER Jobs       Concurrent nvcc processes (default 8).
.PARAMETER Tiers      Restrict the kernel tiers, e.g. "5120" -- cuts a rebuild from ~8 min to
                      ~40 s.  Empty (default) = every tier.
.PARAMETER FullRebuild  Delete the CUDA objects first (forces a clean kernel compile).
.PARAMETER EnablePrac  Include experimental resident/PRAC kernels (default OFF).
.PARAMETER RelinkOnly   Skip the parallel compile and only run the CMake build (host + link).
.PARAMETER Gmp / OpenSslRoot  Override the dependency paths when they are not the defaults.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_stage1_local.ps1
.EXAMPLE
    # fast iteration on one tier while testing a kernel change
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_stage1_local.ps1 -Tiers 5120 -FullRebuild
#>
param(
    [string]$BuildDir = "build_cuda_cmake",
    [string]$Arch = "89",
    [int]$Jobs = 8,
    [string]$Tiers = "",
    [switch]$EnablePrac,       # experimental resident/PRAC kernels; excluded by default
    [switch]$FullRebuild,
    [switch]$RelinkOnly,
    [switch]$Incremental,      # skip CUDA TUs whose object is newer than the sources (fast host-only edits)
    [string]$Extra = "",       # extra cmake definitions, e.g. -Extra "-DECM_MERS_FOLD=1"
    [string]$Gmp = "",
    [string]$OpenSslRoot = "",
    [string]$VcVars = "",
    [string]$CudaRoot = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not [System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
if (-not $Gmp) { $Gmp = Join-Path $repo "third_party\gmp-zen3\dist" }
if (-not $OpenSslRoot) { $OpenSslRoot = "D:/code/vcpkg/installed/x64-windows" }
. (Join-Path $PSScriptRoot 'internal/cuda_toolchain.ps1')
$cuda = Resolve-EcmCudaToolkit -Archs $Arch -CudaRoot $CudaRoot
$vc = Resolve-EcmVcEnvironment $cuda -VcVars $VcVars
$VcVars = $vc.VcVars
Assert-EcmCmakeToolchain $BuildDir $cuda $vc

$cfg = @(
    "-S", "`"$repo`"", "-B", "`"$BuildDir`"", '-G', '"NMake Makefiles"',
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CUDA_COMPILER=`"$($cuda.Nvcc.Replace('\','/'))`"",
    "-DCMAKE_CUDA_HOST_COMPILER=`"$($vc.Compiler.Replace('\','/'))`"",
    "-DCMAKE_CXX_COMPILER=`"$($vc.Compiler.Replace('\','/'))`"",
    "-DECM_ENABLE_CUDA=ON",
    "-DECM_CUDA_FULL_BUILD=ON",
    "-DECM_CUDA_ENABLE_PRAC=$(if ($EnablePrac) { 'ON' } else { 'OFF' })",
    "-DECM_CUDA_ARCHITECTURES=$Arch",
    "-DECM_CUDA_EMBED_PTX=OFF",
    "-DECM_CUDA_PTX_ARCH=",
    "-DECM_CUDA_COMPRESS=ON",
    "-DECM_TPB=128", "-DECM_MAX_ROTATION=1",
    "-DECM_MAXRREG=0", "-DECM_MAXRREG_SMALL=0", "-DECM_MAXRREG_SUYAMA=0",
    "-DECM_REG_TARGET_FORCE=0", "-DECM_NO_PARAM2=0",
    "-DECM_MERS_FOLD=0", "-DECM_PROBE_ADD_DENSITY=1", "-DECM_PROBE_CHAIN_W=0",
    "-DECM_TIERS=$Tiers",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    "-DGMP_INCLUDE_DIR=$($Gmp -replace '\\','/')/include",
    "-DECM_WINDOWS_GMP_ROOT=$($Gmp -replace '\\','/')",
    "-DGMP_LIBRARY=$($Gmp -replace '\\','/')/lib/gmp.lib",
    "-DOPENSSL_ROOT_DIR=$OpenSslRoot"
) -join ' '
if ($Extra) { $cfg = "$cfg $Extra" }

Write-Host "== LOCAL build rule: sm_$Arch, compression ON, no PTX, tiers='$(if ($Tiers) { $Tiers } else { 'all' })' =="
Write-Host "   Experimental resident/PRAC: $([bool]$EnablePrac)"
Write-Host "   CUDA: $($cuda.Root); MSVC: $($vc.Version)"
# nvcc reads a BOM-less source file as GBK; a CJK comment then swallows the next line of
# code (that silently removed `#define CHECKPOINT_VERSION` once).  tools/diag/ensure_bom.ps1
# restores the BOM state git HEAD has, for every tracked source under kernels/ and src/.
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "..\diag\ensure_bom.ps1")
if ($LASTEXITCODE -ne 0) { throw "source encoding check failed" }
& cmd.exe /c "$($vc.Setup) && cmake $cfg" | Select-String -Pattern 'device code|PTX embedded|no PTX|Error|MERSENNE' |
    ForEach-Object { "   $($_.Line.Trim())" }
if ($LASTEXITCODE -ne 0) { throw "configure failed" }

if ($FullRebuild) {
    Get-ChildItem (Join-Path $BuildDir "CMakeFiles\ecm_cuda.dir\kernels\cuda") -Filter *.obj -ErrorAction SilentlyContinue |
        Remove-Item -Force
    Write-Host "   (CUDA objects removed)"
}

if (-not $RelinkOnly) {
    $engineArgs = @('-BuildDir', $BuildDir, '-Jobs', $Jobs, '-VcVars', $VcVars, '-VcVersion', $vc.Version)
    if ($Incremental) { $engineArgs += '-SkipUpToDate' }
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "internal\parallel_nvcc.ps1") @engineArgs
} else {
    & cmd.exe /c "$($vc.Setup) && cmake --build `"$BuildDir`" --target ecm_cuda"
}
if ($LASTEXITCODE -ne 0) { throw "Stage1 build failed ($LASTEXITCODE)" }
Copy-Item -LiteralPath (Join-Path $Gmp 'bin/gmp-10.dll') -Destination $BuildDir -Force

$exe = Join-Path $BuildDir "ecm_cuda.exe"
if (Test-Path $exe) {
    $cuobjdump = Join-Path $cuda.Root "bin\cuobjdump.exe"
    $nCubins = ((& $cuobjdump --list-elf $exe 2>&1) | Select-String '\.cubin').Count
    $nPtx = ((& $cuobjdump --list-ptx $exe 2>&1) | Select-String '\.ptx').Count
    "   ecm_cuda.exe: {0:N0} B ({1:N1} MB)   cubin entries: {2}   ptx entries: {3}" -f `
        (Get-Item $exe).Length, ((Get-Item $exe).Length / 1MB), $nCubins, $nPtx
} else {
    throw "no ecm_cuda.exe in $BuildDir"
}
