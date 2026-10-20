#Requires -Version 5.1
# Experimental target; compile complete modules with recursive source identity.
param([string]$Build='build_cuda_cmake',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89',
      [switch]$Rebuild)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_experiment_probe.ps1')
. (Join-Path $PSScriptRoot '../internal/cuda_build_state.ps1')
$inventory=Get-EcmCudaInventory $repo
$sources=@($inventory.groups.EXPERIMENT_NTT)
Invoke-EcmExperimentBuild -Repo $repo -Build $Build -Arch $Arch -Sources $sources -Name 'ntt_poly_probe' -Rebuild:$Rebuild
