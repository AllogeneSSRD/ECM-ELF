#Requires -Version 5.1
param([string]$Build='build_dev/stage2_point_coop_seed_probe',[string]$Arch='sm_89',
      [ValidateSet(160,192,224,256)][int]$Words=256,[ValidateRange(1,64)][int]$SplitCompile=8)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_experiment_probe.ps1')
Invoke-EcmExperimentBuild -Repo $repo -Build $Build -Arch $Arch `
    -Sources @('tools/test/stage2_point_coop_seed_probe.cu') -Name 'stage2_point_coop_seed_probe' `
    -Defines @("ECM_SEED_WORDS=$Words") -SplitCompile $SplitCompile
