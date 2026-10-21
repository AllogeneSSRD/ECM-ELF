#Requires -Version 5.1
param([string]$BuildDir='build_gui_v2_local', [string]$VcVars='', [string]$VulkanRoot='', [string]$BackendRoot='')
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '../test/build_gui_v2.ps1') -BuildDir $BuildDir -VcVars $VcVars -VulkanRoot $VulkanRoot -BackendRoot $BackendRoot
exit $LASTEXITCODE
