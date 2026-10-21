#Requires -Version 5.1
<#
.SYNOPSIS
Internal Stage1 release worker for one CUDA toolkit group.
.DESCRIPTION
Called by build_stage1_release.ps1. Compiles, splits and packages only the
architectures supported by the selected toolkit. Manifests stay in BuildDir.
Experimental resident/PRAC is opt-in; PtxArch empty means SASS only.
#>
param(
    [string]$BuildDir = "",
    [string]$BuildRoot = "build_cuda_release",
    [string]$OutDir = "dist\cuda",
    [ValidatePattern('^[0-9]+(,[0-9]+)*$')][string]$Archs = "75,86,89,120",
    [ValidatePattern('^$|^[0-9]+$')][string]$PtxArch = "",
    [string]$Tiers = "",
    [switch]$EnablePrac,
    [ValidateRange(1,64)][int]$Jobs = 8,
    [switch]$SkipBuild,
    [switch]$SkipSplit,
    [switch]$SkipPackage,
    [switch]$FullRebuild,
    [string]$Gmp = "",
    [string]$OpenSslRoot = "",
    [string]$VcVars = "",
    [string]$CudaRoot = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
. (Join-Path $PSScriptRoot 'cuda_toolchain.ps1')
$cuda = Resolve-EcmCudaToolkit -Archs $Archs -CudaRoot $CudaRoot
$vc = Resolve-EcmVcEnvironment $cuda -VcVars $VcVars
$VcVars = $vc.VcVars
& cmake "-DECM_CONFIG_ROOT=$repo" -P (Join-Path $PSScriptRoot 'check_ecm_config.cmake')
if ($LASTEXITCODE -ne 0) { throw 'Generated ECM config is stale' }
$archList = @($Archs.Split(','))
if (@($archList | Select-Object -Unique).Count -ne $archList.Count -or ($PtxArch -and $PtxArch -notin $archList)) {
    throw 'Architectures must be unique and include PtxArch'
}
if (-not [System.IO.Path]::IsPathRooted($OutDir)) { $OutDir = Join-Path $repo $OutDir }
. (Join-Path $PSScriptRoot 'gmp_dependency.ps1')
$gmpDependency = Resolve-EcmGmp -Repo $repo -Gmp $Gmp -Release
$Gmp = $gmpDependency.Root
$OutDir = [IO.Path]::GetFullPath($OutDir)
if (-not [IO.Path]::IsPathRooted($Gmp)) { $Gmp = Join-Path $repo $Gmp }
$Gmp = [IO.Path]::GetFullPath($Gmp)
Write-Host "Release GMP: $Gmp (matching header/import library/runtime)"
if (-not $OpenSslRoot) { $OpenSslRoot = "D:/code/vcpkg/installed/x64-windows" }
. (Join-Path $PSScriptRoot 'stage1_release_sources.ps1')
$sourceHashes = Get-Stage1ReleaseSources $repo -EnablePrac:$EnablePrac
$toolkit = $cuda.VersionText
$dependencyHashes = [ordered]@{}
foreach ($name in @('include/gmp.h','lib/gmp.lib','bin/gmp-10.dll')) {
    $dependencyHashes[$name] = (Get-FileHash -LiteralPath (Join-Path $Gmp $name)).Hash
}

    $configureOptions = @(
        '-G', '"NMake Makefiles"',
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_COMPILER=`"$($cuda.Nvcc.Replace('\','/'))`"",
        "-DCMAKE_CUDA_HOST_COMPILER=`"$($vc.Compiler.Replace('\','/'))`"",
        "-DCMAKE_CXX_COMPILER=`"$($vc.Compiler.Replace('\','/'))`"",
        "-DECM_ENABLE_CUDA=ON",
        "-DECM_CUDA_FULL_BUILD=ON",
        "-DECM_CUDA_ENABLE_PRAC=$(if ($EnablePrac) { 'ON' } else { 'OFF' })",
        "-DECM_CUDA_ARCHITECTURES=`"$($Archs -replace ',', ';')`"",
        "-DECM_CUDA_PTX_ARCH=$PtxArch",
        "-DECM_CUDA_EMBED_PTX=OFF",
        "-DECM_CUDA_COMPRESS=OFF",
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
. (Join-Path $PSScriptRoot 'cuda_build_profile.ps1')
$buildProfile = New-EcmCudaProfile $repo release stage1 production $Archs $cuda $vc $gmpDependency ([ordered]@{
    configure = $configureOptions; tiers = $Tiers; enable_prac = $(if ($EnablePrac) { '1' } else { '0' }); extra = '' })
$rootArgument = if ($BuildDir) { '' } else { $BuildRoot }
$BuildDir = Resolve-EcmBuildDirectory $repo $buildProfile $BuildDir $rootArgument
Assert-EcmCmakeToolchain $BuildDir $cuda $vc
Assert-EcmSeparatePackageDirectory $BuildDir $OutDir
Assert-EcmBuildProfile $BuildDir $buildProfile -ReadOnly:$SkipBuild
$manifestPath = Join-Path $BuildDir 'stage1_build_manifest.json'
$cfg = "-S `"$repo`" -B `"$BuildDir`" $configureOptions"
Write-Host "Release leaf: $BuildDir (profile=$($buildProfile.Key))"

if (-not $SkipBuild) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot '../../diag/ensure_bom.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Source encoding check failed' }

    $ptxLabel = if ($PtxArch) { "PTX for sm_$PtxArch only" } else { 'SASS only' }
    Write-Host "== RELEASE rule: archs [$Archs], $ptxLabel, uncompressed (prunable) =="
    Write-Host "   Experimental resident/PRAC: $([bool]$EnablePrac)"
    & cmd.exe /c "$($vc.Setup) && cmake $cfg" | Select-String -Pattern 'device code|PTX embedded|Error' |
        ForEach-Object { "   $($_.Line.Trim())" }
    if ($LASTEXITCODE -ne 0) { throw "configure failed" }
    # Configure and the encoding guard may update generated sources or tracked
    # BOMs. Freeze the actual target closure before compiling.
    $sourceHashes = Get-Stage1ReleaseSources $repo -EnablePrac:$EnablePrac -BuildDir $BuildDir

    if ($FullRebuild) {
        Remove-EcmCudaObjects $BuildDir
        Write-Host "   (CUDA objects removed)"
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "parallel_nvcc.ps1") `
        -BuildDir $BuildDir -Jobs $Jobs -VcVars $VcVars -VcVersion $vc.Version
    if ($LASTEXITCODE -ne 0) { throw "Stage1 release build failed ($LASTEXITCODE)" }
    Copy-Item -LiteralPath (Join-Path $Gmp 'bin/gmp-10.dll') -Destination $BuildDir -Force
    $frozenSources = $sourceHashes | ConvertTo-Json -Depth 3 | ConvertFrom-Json
    Assert-Stage1ReleaseSources $frozenSources (Get-Stage1ReleaseSources $repo -EnablePrac:$EnablePrac -BuildDir $BuildDir)
    foreach ($name in $dependencyHashes.Keys) {
        if ($dependencyHashes[$name] -ne (Get-FileHash -LiteralPath (Join-Path $Gmp $name)).Hash) { throw "Dependency changed during build: $name" }
    }
    $objectHashes = [ordered]@{}
    foreach ($obj in (Get-Stage1CudaObjects $BuildDir)) {
        $objectHashes[$obj] = (Get-FileHash -LiteralPath (Join-Path $BuildDir $obj)).Hash
    }
    [ordered]@{ identity_schema = 1; role = 'release'; profile_sha256 = $buildProfile.Hash;
        source_inventory_sha256 = (Get-FileHash (Join-Path $PSScriptRoot 'cuda_sources.cmake')).Hash;
        compile_setup = $vc.Setup;
        architectures = $Archs; ptx_architecture = $PtxArch; tiers = $Tiers; jobs = $Jobs; enable_prac = [bool]$EnablePrac;
        toolkit = $toolkit; cuda_root = $cuda.Root; host_compiler = $vc.Compiler; host_version = $vc.Version;
        gmp = $Gmp; dependency_hashes = $dependencyHashes;
        source_hashes = $sourceHashes; objects = $objectHashes; executable_sha256 = (Get-FileHash -LiteralPath (Join-Path $BuildDir 'ecm_cuda.exe')).Hash;
        tests_run = $false } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
} else {
    $sourceHashes = Get-Stage1ReleaseSources $repo -EnablePrac:$EnablePrac -BuildDir $BuildDir
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.role -ne 'release' -or $manifest.profile_sha256 -ne $buildProfile.Hash -or
        $manifest.architectures -ne $Archs -or $manifest.ptx_architecture -ne $PtxArch -or $manifest.tiers -ne $Tiers -or
        $manifest.enable_prac -ne [bool]$EnablePrac -or $manifest.toolkit -ne $toolkit -or
        $manifest.cuda_root -ne $cuda.Root -or $manifest.host_compiler -ne $vc.Compiler -or $manifest.host_version -ne $vc.Version -or
        $manifest.executable_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $BuildDir 'ecm_cuda.exe')).Hash) {
        throw 'Stage1 SkipBuild requires the same architecture, tiers, PRAC flag, toolkit and executable'
    }
    Assert-Stage1ReleaseSources $manifest.source_hashes $sourceHashes
    Assert-Stage1BuildIdentity $repo $BuildDir $manifest
    foreach ($name in $dependencyHashes.Keys) {
        if ($manifest.dependency_hashes.$name -ne $dependencyHashes[$name]) { throw "Stage1 dependency changed: $name" }
    }
}

if (-not $SkipSplit) {
    Write-Host "== splitting into one exe per architecture =="
    $splitArgs = @('-BuildDir',$BuildDir,'-OutDir',$OutDir,'-Archs',$Archs,'-VcVars',$VcVars,'-CudaRoot',$cuda.Root)
    if ($PtxArch) { $splitArgs += @('-PtxArch',$PtxArch) }
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "split_stage1_release.ps1") @splitArgs
    if ($LASTEXITCODE -ne 0) { throw "Stage1 release split failed ($LASTEXITCODE)" }
    if (-not $SkipPackage) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'package_stage1.ps1') `
            -BuildDir $BuildDir -OutDir $OutDir -Archs $Archs -Gmp $Gmp
        if ($LASTEXITCODE -ne 0) { throw "Stage1 release packaging failed ($LASTEXITCODE)" }
    }
    Write-Host "== done: see $OutDir =="
}
