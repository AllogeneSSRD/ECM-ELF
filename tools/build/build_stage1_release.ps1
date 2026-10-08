#Requires -Version 5.1
<#
.SYNOPSIS
Parallel Stage1 release build, toolkit groups, architecture split and ZIP packages.
.DESCRIPTION
sm_60/sm_70 use CUDA 12.6; sm_75/sm_86/sm_89/sm_120 use CUDA 13.3.
Each toolkit has an isolated build directory and matching supported MSVC.
The selected PtxArch package keeps PTX; other packages contain SASS only.
.PARAMETER CudaRoot
Modern toolkit override (default installed CUDA 13.3).
.PARAMETER LegacyCudaRoot
Legacy toolkit override (default installed CUDA 12.6).
.EXAMPLE
.\tools\build\build_stage1_release.bat
.EXAMPLE
.\tools\build\build_stage1_release.bat -Archs 60,70 -PtxArch 60 -Tiers 5120
#>
param(
    [string]$BuildDir = 'build_cuda_release',
    [string]$OutDir = 'dist/cuda',
    [ValidatePattern('^[0-9]+(,[0-9]+)*$')][string]$Archs = '60,70,75,86,89,120',
    [ValidatePattern('^$|^[0-9]+$')][string]$PtxArch = '75',
    [string]$Tiers = '',
    [switch]$EnablePrac,
    [ValidateRange(1,64)][int]$Jobs = 8,
    [switch]$SkipBuild,
    [switch]$SkipSplit,
    [switch]$SkipPackage,
    [switch]$FullRebuild,
    [string]$Gmp = '',
    [string]$OpenSslRoot = '',
    [string]$VcVars = '',
    [string]$CudaRoot = '',
    [string]$LegacyCudaRoot = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $PSScriptRoot 'internal/cuda_toolchain.ps1')
$archList = @($Archs.Split(','))
if (@($archList | Select-Object -Unique).Count -ne $archList.Count -or ($PtxArch -and $PtxArch -notin $archList)) {
    throw 'Architectures must be unique and include PtxArch when PTX is enabled'
}
if (-not [IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
if (-not [IO.Path]::IsPathRooted($OutDir)) { $OutDir = Join-Path $repo $OutDir }
$groups = @()
foreach ($legacy in @($true,$false)) {
    $selected = @($archList | Where-Object { ([int]$_ -lt 75) -eq $legacy })
    if (-not $selected.Count) { continue }
    $root = if ($legacy) { $LegacyCudaRoot } else { $CudaRoot }
    $cuda = Resolve-EcmCudaToolkit -Archs ($selected -join ',') -CudaRoot $root
    $vc = Resolve-EcmVcEnvironment $cuda -VcVars $VcVars
    $groups += [pscustomobject]@{ Archs = $selected; Cuda = $cuda; Vc = $vc; Name = $(if ($legacy) { 'cuda12_6' } else { 'cuda13_3' }) }
}
foreach ($group in $groups) {
    $groupBuild = Join-Path $BuildDir $group.Name
    Write-Host "== Stage1 release [$($group.Archs -join ',')] CUDA=$($group.Cuda.Root) MSVC=$($group.Vc.Version) =="
    $forward = @('-BuildDir',$groupBuild,'-OutDir',$OutDir,'-Archs',($group.Archs -join ','),'-Jobs',$Jobs,
        '-CudaRoot',$group.Cuda.Root,'-VcVars',$group.Vc.VcVars)
    if ($PtxArch -in $group.Archs) { $forward += @('-PtxArch',$PtxArch) }
    foreach ($name in @('Tiers','Gmp','OpenSslRoot')) { if ((Get-Variable -Name $name -ValueOnly)) { $forward += @(('-' + $name),(Get-Variable -Name $name -ValueOnly)) } }
    foreach ($name in @('EnablePrac','SkipBuild','SkipSplit','SkipPackage','FullRebuild')) { if ((Get-Variable -Name $name -ValueOnly)) { $forward += '-' + $name } }
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'internal/build_stage1_release_group.ps1') @forward
    if ($LASTEXITCODE -ne 0) { throw "Stage1 release failed: $($group.Name)" }
}
Write-Host "Stage1 release complete: $OutDir"
