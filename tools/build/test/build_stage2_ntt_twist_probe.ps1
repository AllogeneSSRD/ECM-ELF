param([string]$Build='build_dev/stage2_ntt_twist_probe', [string]$ModuleBuild='',
      [string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_ntt_probe.ps1')
Invoke-EcmNttProbeBuild -Repo $repo -Build $Build -Arch $Arch -ModuleBuild $ModuleBuild `
    -Source 'tools/test/stage2_ntt_twist_probe.cu' -Name 'stage2_ntt_twist_probe'
