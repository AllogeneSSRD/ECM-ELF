#Requires -Version 5.1
param([string]$Build='build_dev/stage2_point_coop_probe',[string]$Arch='sm_89',
      [ValidateSet(4,8,16)][int]$Tpi=16,[ValidateSet(64,128,256)][int]$MaxWords=256,
      [ValidateRange(1,64)][int]$SplitCompile=8,[switch]$Rebuild)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_experiment_probe.ps1')
Invoke-EcmExperimentBuild -Repo $repo -Build $Build -Arch $Arch `
    -Sources @('tools/test/stage2_point_coop_probe.cu') -Name 'stage2_point_coop_probe' `
    -Defines @("ECM_COOP_TPI=$Tpi","ECM_COOP_MAX_WORDS=$MaxWords") -SplitCompile $SplitCompile -Rebuild:$Rebuild
