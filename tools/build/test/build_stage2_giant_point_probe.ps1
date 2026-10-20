#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/stage2_giant_point_probe',[string]$Arch='sm_89',[string]$WorkerBuild='')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot));Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_ntt_probe.ps1')
. (Join-Path $PSScriptRoot '../internal/cuda_build_state.ps1')
$inventory=Get-EcmCudaInventory $repo
Invoke-EcmNttProbeBuild -Repo $repo -Build $Build -Arch $Arch -Source 'tools/test/stage2_giant_point_probe.cu' `
    -Name 'stage2_giant_point_probe' -ModuleGroup STAGE2_ARITHMETIC -DefaultStream legacy `
    -WorkerBuild $WorkerBuild -ExtraSources @($inventory.groups.EXPERIMENT_GIANT)
