#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/stage2_budget_shape_probe',
      [string]$SourceRoot='build_cuda_cmake/_point_scratch_20261005/native_calibrated/sources',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89')
$ErrorActionPreference='Stop';$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot));Set-Location $repo
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
$SourceRoot=(Resolve-Path $SourceRoot).Path
if(Test-Path (Join-Path $Build 'manifest.json')){throw 'Use a fresh build directory'}
. (Join-Path $PSScriptRoot '../internal/cuda_frozen_ntt_probe.ps1')
$result=Invoke-EcmFrozenNttProbeBuild -Repo $repo -SourceRoot $SourceRoot -Build $Build -Arch $Arch `
    -Source 'tools/test/stage2_budget_shape_probe.cu' -Name 'stage2_budget_shape_probe' -Defines @('NTT_GL_FIXED_MODE=3')
$result.gl_fixed_mode=3
$result|ConvertTo-Json -Depth 8|Set-Content (Join-Path $Build 'manifest.json') -Encoding utf8
Write-Host "built $($result.exe)"
