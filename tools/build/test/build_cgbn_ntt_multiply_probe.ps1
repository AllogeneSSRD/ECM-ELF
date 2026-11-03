# Integer crossover probe linked against the production NTT inventory.
param([string]$Build='build_dev/cgbn_ntt_multiply_probe',[string]$WorkerBuild='',[string]$ModuleBuild='',
      [ValidateSet(512,1024,2048,4096,8192,16384,32768,65536,131072)][int]$Bits=8192,
      [ValidateSet(8,16,32)][int]$Tpi=16,[string]$Arch='sm_89',
      [ValidateRange(1,64)][int]$SplitCompile=8)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot));Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_ntt_probe.ps1')
Invoke-EcmNttProbeBuild -Repo $repo -Build $Build -Arch $Arch -WorkerBuild $WorkerBuild -ModuleBuild $ModuleBuild `
    -Source 'tools/test/cgbn_ntt_multiply_probe.cu' -Name 'cgbn_ntt_multiply_probe' `
    -LocalDefines @("ECM_CROSS_BITS=$Bits","ECM_CROSS_TPI=$Tpi") -SplitCompile $SplitCompile
