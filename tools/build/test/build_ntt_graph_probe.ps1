#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/ntt_graph_probe',[ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89',[string]$WorkerBuild='')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot));Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_ntt_probe.ps1')
Invoke-EcmNttProbeBuild -Repo $repo -Build $Build -Arch $Arch -Source 'tools/test/ntt_graph_probe.cu' -Name 'ntt_graph_probe' -DefaultStream 'legacy' -WorkerBuild $WorkerBuild
