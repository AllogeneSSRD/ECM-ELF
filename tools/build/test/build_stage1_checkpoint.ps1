#Requires -Version 5.1
# Isolated Stage1 checkpoint/naming regression build; never deploys to production.
param([string]$Build='build_cuda_cmake/stage1_checkpoint_v2_20261017',
      [string]$Tiers='1024',[int]$Jobs=6,[switch]$RelinkOnly)
$ErrorActionPreference='Stop'
$options=@{Stage='stage1';Build=$Build;Tiers=$Tiers;Jobs=$Jobs}
if($RelinkOnly){$options.RelinkOnly=$true}
& (Join-Path $PSScriptRoot 'build_edwards_gpu_workers.ps1') @options
if($LASTEXITCODE -ne 0){throw 'checkpoint test build failed'}
