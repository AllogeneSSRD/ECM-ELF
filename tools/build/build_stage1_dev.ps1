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
    [string]$BuildDir = '',
    [string]$Arch = '89',
    [int]$Jobs = 8,
    [string]$Tiers = '',
    [ValidateSet('isolated','grouped')][string]$KernelLayout = 'isolated',
    [switch]$EnablePrac,
    [switch]$FullRebuild,
    [switch]$RelinkOnly,
    [switch]$HostOnly,
    [switch]$Incremental = $true,
    [string]$Extra = '',
    [string]$Gmp = '',
    [string]$OpenSslRoot = '',
    [string]$VcVars = '',
    [string]$CudaRoot = ''
)
$ErrorActionPreference = 'Stop'
# Forward only supplied options; local owns arithmetic and compiler defaults.
$PSBoundParameters['BuildDir'] = $BuildDir
$PSBoundParameters['BuildRole'] = 'dev'
& (Join-Path $PSScriptRoot 'build_stage1_local.ps1') @PSBoundParameters
