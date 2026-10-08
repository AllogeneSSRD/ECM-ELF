#Requires -Version 5.1
<#
.SYNOPSIS
Build Stage1 in an isolated development directory, with the same safe defaults as local.
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage1_dev.ps1 -Tiers 4608 -Incremental
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage1_dev.ps1 -Extra "-DECM_TPB=256"
#>
param(
    [string]$BuildDir = 'build_cuda_dev',
    [string]$Arch = '89',
    [int]$Jobs = 8,
    [string]$Tiers = '',
    [switch]$EnablePrac,
    [switch]$FullRebuild,
    [switch]$RelinkOnly,
    [switch]$Incremental,
    [string]$Extra = '',
    [string]$Gmp = '',
    [string]$OpenSslRoot = '',
    [string]$VcVars = '',
    [string]$CudaRoot = ''
)
$ErrorActionPreference = 'Stop'
# Forward only supplied options; local owns arithmetic and compiler defaults.
$PSBoundParameters['BuildDir'] = $BuildDir
& (Join-Path $PSScriptRoot 'build_stage1_local.ps1') @PSBoundParameters
