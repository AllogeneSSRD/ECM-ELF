param([string]$Build='build_dev/stage1_mersenne_ntt', [string]$ModuleBuild='',
      [string]$Arch='sm_89', [ValidateRange(1,64)][int]$SplitCompile=1)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_ntt_probe.ps1')
Invoke-EcmNttProbeBuild -Repo $repo -Build $Build -Arch $Arch -ModuleBuild $ModuleBuild `
    -Source 'tools/test/stage1_mersenne_ntt.cpp' -Name 'stage1_mersenne_ntt' `
    -ExtraSources @('src/ecm/stage1/montgomery/cuda/ntt/ladder.cu','src/ecm/stage1/scalar.cpp') `
    -SplitCompile $SplitCompile
