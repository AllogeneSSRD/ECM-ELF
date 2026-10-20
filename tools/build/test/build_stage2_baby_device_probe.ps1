#Requires -Version 5.1
# Experimental target; compile complete modules with recursive source identity.
param([string]$Build='build_cuda_cmake/baby_device_probe',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89',
      [switch]$Rebuild)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_experiment_probe.ps1')
$sources=@('tools/test/stage2_baby_device_probe.cu')
Invoke-EcmExperimentBuild -Repo $repo -Build $Build -Arch $Arch -Sources $sources -Name 'stage2_baby_device_probe' -Rebuild:$Rebuild
