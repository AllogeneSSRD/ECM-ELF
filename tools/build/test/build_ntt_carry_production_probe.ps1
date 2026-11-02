# Exact carry gates and operator comparisons against the actual production modules.
param([string]$Build='build_dev/ntt_carry_production_probe',[string]$WorkerBuild='build_stage2_local',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_ntt_probe.ps1')
Invoke-EcmNttProbeBuild -Repo $repo -Build $Build -Arch $Arch -WorkerBuild $WorkerBuild `
    -Source 'tools/test/ntt_carry_partial_probe.cu' -Name 'ntt_carry_production_probe' -Defines @('ECM_CARRY_PRODUCTION=1')
