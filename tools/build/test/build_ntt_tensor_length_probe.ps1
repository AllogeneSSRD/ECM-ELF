#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/ntt_tensor_length_probe',[ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot));Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_ntt_probe.ps1')
Invoke-EcmNttProbeBuild -Repo $repo -Build $Build -Arch $Arch -Source 'tools/bench/ntt_tensor_length_probe.cu' -Name 'ntt_tensor_length_probe' -DefaultStream 'per-thread'
