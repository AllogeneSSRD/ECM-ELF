#Requires -Version 5.1
<#
.SYNOPSIS
Parallel-compile and package production Stage2 for one or more GPU architectures.
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage2_release.ps1
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage2_release.ps1 -Arch sm_89
#>
param(
    [ValidatePattern('^sm_[0-9]+$')][string]$Arch = 'sm_89',
    [ValidatePattern('^[0-9]+(,[0-9]+)*$')][string]$Archs = '60,70,75,86,89,120',
    [string]$Build = '',
    [string]$OutDir = '',
    [ValidateRange(1,64)][int]$SplitCompile = 8,
    [string]$Gmp = '',
    [string]$Stage1Exe = '',
    [string]$CudaRoot = '',
    [string]$LegacyCudaRoot = '',
    [string]$VcVars = '',
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $PSScriptRoot 'internal/gmp_dependency.ps1')
$gmpDependency = Resolve-EcmGmp -Repo $repo -Gmp $Gmp -Release
$Gmp = $gmpDependency.Root
Write-Host "Release GMP: $Gmp (matching header/import library/runtime)"
if ($PSBoundParameters.ContainsKey('Arch')) {
    if ($PSBoundParameters.ContainsKey('Archs')) { throw 'Use either -Arch or -Archs' }
}
$architectures = @(if ($PSBoundParameters.ContainsKey('Arch')) { $Arch } else { $Archs.Split(',') | ForEach-Object { 'sm_' + $_ } })
if (@($architectures | Select-Object -Unique).Count -ne $architectures.Count) { throw 'Duplicate architecture' }
. (Join-Path $PSScriptRoot 'internal/cuda_toolchain.ps1')
$toolchains = @{}
foreach ($target in $architectures) {
    $root = if ([int]$target.Substring(3) -lt 75) { $LegacyCudaRoot } else { $CudaRoot }
    $cuda = Resolve-EcmCudaToolkit -Archs $target -CudaRoot $root
    $vc = Resolve-EcmVcEnvironment $cuda -VcVars $VcVars
    $toolchains[$target] = [pscustomobject]@{ Cuda = $cuda; Vc = $vc }
}
$multi = $architectures.Count -gt 1
if (-not $Build) { $Build = if ($multi) { 'build_cuda_cmake/release_stage2' } else { 'build_cuda_cmake/release_ux_' + $architectures[0].Replace('_','') } }
if (-not $OutDir) { $OutDir = if ($multi) { 'dist/cuda-stage2' } else { 'dist/cuda-stage2-' + $architectures[0].Replace('_','') } }
foreach ($target in $architectures) {
    $targetBuild = if ($multi) { Join-Path $Build $target } else { $Build }
    $targetOut = if ($multi) { Join-Path $OutDir ('cuda-stage2-' + $target.Replace('_','')) } else { $OutDir }
    $tools = $toolchains[$target]
    Write-Host "== Stage2 release $target (split compile=$SplitCompile) CUDA=$($tools.Cuda.Root) MSVC=$($tools.Vc.Version) =="
    $argsForPackage = @('-Arch', $target, '-Build', $targetBuild, '-OutDir', $targetOut, '-SplitCompile', $SplitCompile, '-Gmp', $Gmp,
        '-CudaRoot',$tools.Cuda.Root,'-VcVars',$tools.Vc.VcVars)
    if ($Stage1Exe) { $argsForPackage += @('-Stage1Exe', $Stage1Exe) }
    if ($SkipBuild) { $argsForPackage += '-SkipBuild' }
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'internal/package_stage2.ps1') @argsForPackage
    if ($LASTEXITCODE -ne 0) { throw "Stage2 release failed: $target" }
}
Write-Host "Stage2 release complete: $OutDir"
