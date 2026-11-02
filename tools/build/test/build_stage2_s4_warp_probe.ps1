#Requires -Version 5.1
param([string]$Build='build_dev/stage2_s4_warp_probe',[string]$Arch='sm_89',
      [ValidateRange(1,64)][int]$SplitCompile=1)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_experiment_probe.ps1')
Invoke-EcmExperimentBuild -Repo $repo -Build $Build -Arch $Arch `
    -Sources @('tools/test/stage2_s4_warp_probe.cu') -Name 'stage2_s4_warp_probe' -SplitCompile $SplitCompile
