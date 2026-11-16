#Requires -Version 5.1
param([string]$Build='data/experiments/stage2_baby_wheel_06322/build',
      [string]$Arch='sm_89',[ValidateRange(1,64)][int]$SplitCompile=8,[switch]$Rebuild)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_experiment_probe.ps1')
Invoke-EcmExperimentBuild -Repo $repo -Build $Build -Arch $Arch `
    -Sources @('tools/test/stage2_baby_wheel_probe.cu') -Name 'stage2_baby_wheel_probe' `
    -SplitCompile $SplitCompile -Rebuild:$Rebuild
