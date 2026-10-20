# Isolated native build entry for the CUDA source/dependency identity milestone.
param([ValidateSet('stage1','stage2','development')][string]$Stage = 'stage1',
      [string]$Build = 'build_cuda_cmake/cuda_structure_p0',
      [string]$Tiers = '384', [int]$Jobs = 8,
      [ValidateRange(1,64)][int]$SplitCompile = 8)
$ErrorActionPreference = 'Stop'
$buildRoot = Split-Path -Parent $PSScriptRoot
if ($Stage -eq 'stage1') {
    & (Join-Path $PSScriptRoot 'build_edwards_gpu_workers.ps1') -Stage stage1 -Build $Build -Tiers $Tiers -Jobs $Jobs
} else {
    $engine = if ($Stage -eq 'stage2') { 'production' } else { 'development' }
    & (Join-Path $buildRoot 'build_stage2_local.ps1') -Build (Join-Path $Build $Stage) -Engine $engine -SplitCompile $SplitCompile -GlBackend ptx -AddSubMask 1 -OuterUnrollU 0
}
if ($LASTEXITCODE -ne 0) { throw "P0 native build failed: $Stage" }
