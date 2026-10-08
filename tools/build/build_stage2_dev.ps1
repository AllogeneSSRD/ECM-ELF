#Requires -Version 5.1
<#
.SYNOPSIS
Build the experimental Stage2 queue/save executable in an isolated directory.
.DESCRIPTION
Uses tools/bench/ecm_cuda_stage2_dev.cu. Arithmetic defaults match production;
alternative backends and schedules require explicit parameters. For the lower-level
tree harness, use dev/build_stage2_tree_gpu.ps1.
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage2_dev.ps1
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage2_dev.ps1 -AddSubMask 0 -Rebuild
#>
param(
    [string]$Build = 'build_cuda_cmake/development_stage2',
    [ValidatePattern('^sm_[0-9]+$')][string]$Arch = 'sm_89',
    [ValidateSet('runtime','fold','short','ptx')][string]$GlBackend = 'ptx',
    [ValidateSet(0,4)][int]$OuterUnrollU = 0,
    [ValidateSet(0,1,2,3)][int]$AddSubMask = 1,
    [ValidateRange(1,64)][int]$SplitCompile = 8,
    [string]$Gmp = '',
    [string]$CudaRoot = '',
    [string]$VcVars = '',
    [switch]$HostOnly,
    [switch]$Rebuild
)
$ErrorActionPreference = 'Stop'
$PSBoundParameters['Build'] = $Build
$PSBoundParameters['Engine'] = 'development'
$PSBoundParameters['AddSubMask'] = $AddSubMask
& (Join-Path $PSScriptRoot 'build_stage2_local.ps1') @PSBoundParameters
