<#
.SYNOPSIS
    Local Stage1 build for ecm_cuda, using the general Montgomery baseline.

.DESCRIPTION
    Uses the fixed build_stage1_local directory by default.
    A different profile requires an explicit BuildDir; existing objects are never relabelled.
    Explicit BuildDir remains a final leaf. Defaults are in docs/usage/BUILD.md:

        compression ON      -Xfatbin -compress-all
        no PTX              ECM_CUDA_EMBED_PTX=OFF        (needs exactly sm_<Arch>)
        one architecture    sm_89                         (override with -Arch)
        full kernel set, all tiers, param2 enabled
        isolated curve/parameterization/tier device modules

    The compile step uses tools\build\internal\parallel_nvcc.ps1 for independent CUDA TUs.

.PARAMETER BuildDir   Explicit final build directory; omitted uses build_stage1_local.
.PARAMETER Arch       CUDA architecture (default 89).
.PARAMETER Jobs       Concurrent nvcc processes (default 8).
.PARAMETER Tiers      Restrict the kernel tiers, e.g. "5120". Empty (default) = every tier.
.PARAMETER FullRebuild  Delete the CUDA objects first (forces a clean kernel compile).
.PARAMETER EnablePrac  Include experimental resident/PRAC kernels (default OFF).
.PARAMETER KernelLayout  Isolated modules by default; grouped retains the aggregate source layout.
.PARAMETER HostOnly    Verify all CUDA objects and refuse any CUDA compilation.
.PARAMETER RelinkOnly  Removed unsafe bypass; use verified reuse or HostOnly instead.
.PARAMETER Gmp / OpenSslRoot  Override the dependency paths when they are not the defaults.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_stage1_local.ps1
.EXAMPLE
    # fast iteration on one tier while testing a kernel change
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_stage1_local.ps1 -Tiers 5120 -FullRebuild
#>
param(
    [string]$BuildDir = "",
    [ValidateSet('local','dev')][string]$BuildRole = 'local',
    [string]$Arch = "89",
    [int]$Jobs = 8,
    [string]$Tiers = "",
    [ValidateSet('isolated','grouped')][string]$KernelLayout = 'isolated',
    [switch]$EnablePrac,       # experimental resident/PRAC kernels; excluded by default
    [switch]$FullRebuild,
    [switch]$RelinkOnly,
    [switch]$HostOnly,
    [switch]$Incremental = $true, # default: verify dependency/command/object identities before reuse
    [string]$Extra = "",       # extra cmake definitions, e.g. -Extra "-DECM_MERS_FOLD=1"
    [string]$Gmp = "",
    [string]$OpenSslRoot = "",
    [string]$VcVars = "",
    [string]$CudaRoot = ""
)

$ErrorActionPreference = "Stop"
if ($RelinkOnly) { throw 'RelinkOnly no longer bypasses verification; use normal reuse or HostOnly for host edits' }
if ($HostOnly -and ($FullRebuild -or -not $Incremental)) { throw 'HostOnly and forced CUDA rebuild are mutually exclusive' }
if ($Extra -match '(?i)(?:^|\s)-(?:S|B|G)(?:\s|$)|-D(?:CMAKE_(?:CUDA_COMPILER|CUDA_HOST_COMPILER|CXX_COMPILER|BUILD_TYPE)|ECM_CUDA_(?:ARCHITECTURES|PTX_ARCH|EMBED_PTX|COMPRESS|ENABLE_PRAC|ISOLATE_TIERS)|ECM_TIERS)(?::\w+)?=') {
    throw 'Extra cannot override toolchain, architecture, output or named build policy; use the corresponding build parameters'
}
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $PSScriptRoot 'internal/gmp_dependency.ps1')
$gmpDependency = Resolve-EcmGmp -Repo $repo -Gmp $Gmp
$Gmp = $gmpDependency.Root
if (-not $OpenSslRoot) { $OpenSslRoot = "D:/code/vcpkg/installed/x64-windows" }
. (Join-Path $PSScriptRoot 'internal/cuda_toolchain.ps1')
. (Join-Path $PSScriptRoot 'internal/cuda_build_state.ps1')
. (Join-Path $PSScriptRoot 'internal/cuda_build_profile.ps1')
$cuda = Resolve-EcmCudaToolkit -Archs $Arch -CudaRoot $CudaRoot
$vc = Resolve-EcmVcEnvironment $cuda -VcVars $VcVars
$VcVars = $vc.VcVars
$configureOptions = @(
    '-G', '"NMake Makefiles"',
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CUDA_COMPILER=`"$($cuda.Nvcc.Replace('\','/'))`"",
    "-DCMAKE_CUDA_HOST_COMPILER=`"$($vc.Compiler.Replace('\','/'))`"",
    "-DCMAKE_CXX_COMPILER=`"$($vc.Compiler.Replace('\','/'))`"",
    "-DECM_ENABLE_CUDA=ON",
    "-DECM_CUDA_ISOLATE_TIERS=$(if ($KernelLayout -eq 'isolated') { 'ON' } else { 'OFF' })",
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
if ($Extra) { $configureOptions = "$configureOptions $Extra" }
$buildProfile = New-EcmCudaProfile $repo $BuildRole stage1 production $Arch $cuda $vc $gmpDependency ([ordered]@{
    configure = $configureOptions; tiers = $Tiers; enable_prac = $(if ($EnablePrac) { '1' } else { '0' }); extra = $Extra })
$BuildDir = Resolve-EcmBuildDirectory $repo $buildProfile $BuildDir
Assert-EcmCmakeToolchain $BuildDir $cuda $vc
if ($HostOnly -and (-not (Test-Path -LiteralPath (Join-Path $BuildDir 'compile_commands.json')) -or
    -not (Test-Path -LiteralPath (Join-Path $BuildDir "CMakeFiles/$($buildProfile.Fields.cmake_version)/CMakeCUDACompiler.cmake")))) {
    throw 'HostOnly requires an existing configured build with the same CMake version; run normal reuse first'
}
Assert-EcmBuildProfile $BuildDir $buildProfile
$cfg = "-S `"$repo`" -B `"$BuildDir`" $configureOptions"

Write-Host "== LOCAL build rule: sm_$Arch, compression ON, no PTX, tiers='$(if ($Tiers) { $Tiers } else { 'all' })' =="
Write-Host "   Experimental resident/PRAC: $([bool]$EnablePrac)"
Write-Host "   CUDA module layout: $KernelLayout"
Write-Host "   CUDA: $($cuda.Root); MSVC: $($vc.Version)"
Write-Host "   role=$BuildRole profile=$($buildProfile.Key) leaf=$BuildDir"
# nvcc reads a BOM-less source file as GBK; a CJK comment then swallows the next line of
# code (that silently removed `#define CHECKPOINT_VERSION` once).  tools/diag/ensure_bom.ps1
# restores the BOM state git HEAD has, for every tracked source under kernels/ and src/.
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "..\diag\ensure_bom.ps1")
if ($LASTEXITCODE -ne 0) { throw "source encoding check failed" }
& cmd.exe /c "$($vc.Setup) && cmake $cfg" | Select-String -Pattern 'device code|PTX embedded|no PTX|Error|MERSENNE' |
    ForEach-Object { "   $($_.Line.Trim())" }
if ($LASTEXITCODE -ne 0) { throw "configure failed" }

if ($FullRebuild) {
    Remove-EcmCudaObjects $BuildDir
    Write-Host "   (CUDA objects removed)"
}

$engineArgs = @('-BuildDir', $BuildDir, '-Jobs', $Jobs, '-VcVars', $VcVars, '-VcVersion', $vc.Version)
if (-not $Incremental) { $engineArgs += '-Rebuild' }
if ($HostOnly) { $engineArgs += '-HostOnly' }
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "internal\parallel_nvcc.ps1") @engineArgs
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

. (Join-Path $PSScriptRoot 'internal/stage1_release_sources.ps1')
Assert-EcmBuildProfile $BuildDir $buildProfile -ReadOnly
$localObjects = [ordered]@{}
foreach ($obj in (Get-Stage1CudaObjects $BuildDir)) {
    $localObjects[$obj] = (Get-FileHash -LiteralPath (Join-Path $BuildDir $obj)).Hash
}
[ordered]@{ identity_schema = 1; role = $BuildRole; profile_sha256 = $buildProfile.Hash;
    engine = 'production'; tiers = $Tiers; enable_prac = [bool]$EnablePrac; kernel_layout = $KernelLayout;
    compile_setup = $vc.Setup; gmp_root = $Gmp;
    dependency_hashes = [ordered]@{ header = (Get-FileHash (Join-Path $Gmp 'include/gmp.h')).Hash;
        library = (Get-FileHash (Join-Path $Gmp 'lib/gmp.lib')).Hash; runtime = (Get-FileHash (Join-Path $Gmp 'bin/gmp-10.dll')).Hash };
    architecture = $Arch; cuda_root = $cuda.Root; toolkit = $cuda.VersionText;
    host_compiler = $vc.Compiler; host_version = $vc.Version;
    source_inventory_sha256 = (Get-FileHash (Join-Path $PSScriptRoot 'internal/cuda_sources.cmake')).Hash;
    source_hashes = (Get-Stage1ReleaseSources $repo -EnablePrac:$EnablePrac -BuildDir $BuildDir);
    objects = $localObjects; executable_sha256 = (Get-FileHash -LiteralPath $exe).Hash
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $BuildDir 'stage1_local_manifest.json') -Encoding utf8
