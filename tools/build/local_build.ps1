<#
.SYNOPSIS
    LOCAL test build for ecm_cuda -- the long-term rule for this machine, one command.

.DESCRIPTION
    Configures and builds `build_cuda_cmake` with the settings fixed by the project rule
    (docs/ECM_CGBN_OPTIMIZATION.md 8.9):

        compression ON      -Xfatbin -compress-all        (-62% device-code size, measured)
        no PTX              ECM_CUDA_EMBED_PTX=OFF        (-19% more; needs exactly sm_<Arch>)
        one architecture    sm_89                         (both GPUs of this box are 8.9)
        full kernel set, all tiers, param2 enabled

    Result on this machine: ecm_cuda.exe 70.4 MB -> 10.4 MB.

    The compile step uses tools\build\parallel_nvcc.ps1 (NMake alone is serial: a full kernel
    build is ~40 min serial vs ~8 min with 6 concurrent nvcc).

.PARAMETER BuildDir   Build directory (default build_cuda_cmake, the local test dir).
.PARAMETER Arch       CUDA architecture (default 89).
.PARAMETER Jobs       Concurrent nvcc processes (default 6; more does not help, see 8.8).
.PARAMETER Tiers      Restrict the kernel tiers, e.g. "5120" -- cuts a rebuild from ~8 min to
                      ~40 s.  Empty (default) = every tier.
.PARAMETER FullRebuild  Delete the CUDA objects first (forces a clean kernel compile).
.PARAMETER RelinkOnly   Skip the parallel compile and only run the CMake build (host + link).
.PARAMETER Gmp / OpenSslRoot  Override the dependency paths when they are not the defaults.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\local_build.ps1
.EXAMPLE
    # fast iteration on one tier while testing a kernel change
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\local_build.ps1 -Tiers 5120 -FullRebuild
#>
param(
    [string]$BuildDir = "build_cuda_cmake",
    [string]$Arch = "89",
    [int]$Jobs = 6,
    [string]$Tiers = "",
    [switch]$FullRebuild,
    [switch]$RelinkOnly,
    [switch]$Incremental,      # skip CUDA TUs whose object is newer than the sources (fast host-only edits)
    [string]$Extra = "",       # extra cmake definitions, e.g. -Extra "-DECM_MERS_FOLD=1"
    [string]$Gmp = "",
    [string]$OpenSslRoot = "",
    [string]$VcVars = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not [System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
if (-not $Gmp) { $Gmp = Join-Path $repo "third_party\gmp-zen3\dist" }
if (-not $OpenSslRoot) { $OpenSslRoot = "D:/code/vcpkg/installed/x64-windows" }
if (-not $VcVars) {
    $cand = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending
    if (-not $cand) { throw "vcvars64.bat not found; pass -VcVars" }
    $VcVars = $cand[0].FullName
}

$cfg = @(
    "-S", "`"$repo`"", "-B", "`"$BuildDir`"", '-G', '"NMake Makefiles"',
    "-DCMAKE_BUILD_TYPE=Release",
    "-DECM_ENABLE_CUDA=ON",
    "-DECM_CUDA_FULL_BUILD=ON",
    "-DECM_CUDA_ARCHITECTURES=$Arch",
    "-DECM_CUDA_EMBED_PTX=OFF",
    "-DECM_CUDA_COMPRESS=ON",
    "-DECM_TPB=128", "-DECM_MAX_ROTATION=1",
    "-DECM_MAXRREG_SMALL=0", "-DECM_MAXRREG_SUYAMA=0",
    "-DECM_REG_TARGET_FORCE=0", "-DECM_NO_PARAM2=0",
    "-DECM_TIERS=$Tiers",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    "-DGMP_INCLUDE_DIR=$($Gmp -replace '\\','/')/include",
    "-DGMP_LIBRARY=$($Gmp -replace '\\','/')/lib/gmp.lib",
    "-DOPENSSL_ROOT_DIR=$OpenSslRoot"
) -join ' '
if ($Extra) { $cfg = "$cfg $Extra" }

Write-Host "== LOCAL build rule: sm_$Arch, compression ON, no PTX, tiers='$(if ($Tiers) { $Tiers } else { 'all' })' =="
# nvcc reads a BOM-less source file as GBK; a CJK comment then swallows the next line of
# code (that silently removed `#define CHECKPOINT_VERSION` once).  tools/diag/ensure_bom.ps1
# restores the BOM state git HEAD has, for every tracked source under kernels/ and src/.
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "..\diag\ensure_bom.ps1")
& cmd.exe /c "call `"$VcVars`" >nul 2>&1 && cmake $cfg" | Select-String -Pattern 'device code|PTX embedded|no PTX|Error|MERSENNE' |
    ForEach-Object { "   $($_.Line.Trim())" }
if ($LASTEXITCODE -ne 0) { throw "configure failed" }

if ($FullRebuild) {
    Get-ChildItem (Join-Path $BuildDir "CMakeFiles\ecm_cuda.dir\kernels\cuda") -Filter *.obj -ErrorAction SilentlyContinue |
        Remove-Item -Force
    Write-Host "   (CUDA objects removed)"
}

if (-not $RelinkOnly) {
    $engineArgs = @('-BuildDir', $BuildDir, '-Jobs', $Jobs)
    if ($Incremental) { $engineArgs += '-SkipUpToDate' }
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "parallel_nvcc.ps1") @engineArgs
} else {
    & cmd.exe /c "call `"$VcVars`" >nul 2>&1 && cmake --build `"$BuildDir`" --target ecm_cuda"
}

$exe = Join-Path $BuildDir "ecm_cuda.exe"
if (Test-Path $exe) {
    $cuda = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3" }
    $cuobjdump = Join-Path $cuda "bin\cuobjdump.exe"
    $nCubins = ((& $cuobjdump --list-elf $exe 2>&1) | Select-String '\.cubin').Count
    $nPtx = ((& $cuobjdump --list-ptx $exe 2>&1) | Select-String '\.ptx').Count
    "   ecm_cuda.exe: {0:N0} B ({1:N1} MB)   cubin entries: {2}   ptx entries: {3}" -f `
        (Get-Item $exe).Length, ((Get-Item $exe).Length / 1MB), $nCubins, $nPtx
} else {
    throw "no ecm_cuda.exe in $BuildDir"
}
