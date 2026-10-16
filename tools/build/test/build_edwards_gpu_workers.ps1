#Requires -Version 5.1
param([ValidateSet('stage1','stage2','both')][string]$Stage='both',
      [string]$Build='build_cuda_cmake/edwards_gpu_workers',
      [string]$Arch='89',
      [string]$Tiers='1024',
      [switch]$RelinkOnly)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
if($Stage -ne 'stage2') {
    if($RelinkOnly){
        & (Join-Path $root 'build_stage1_local.ps1') -BuildDir "$Build/stage1" -Arch $Arch -Tiers $Tiers -RelinkOnly
    } else {
        & (Join-Path $root 'build_stage1_local.ps1') -BuildDir "$Build/stage1" -Arch $Arch -Tiers $Tiers -Incremental
    }
    if($LASTEXITCODE -ne 0){throw 'Edwards Stage1 worker build failed'}
}
if($Stage -ne 'stage1') {
    & (Join-Path $root 'build_stage2_local.ps1') -Build "$Build/stage2" -Arch "sm_$Arch" -Engine production
    if($LASTEXITCODE -ne 0){throw 'Edwards Stage2 worker build failed'}
}
