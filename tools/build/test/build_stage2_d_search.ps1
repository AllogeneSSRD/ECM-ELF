#Requires -Version 5.1
param(
    [string]$BaseBuild = 'build_cuda_cmake/local_stage2_0_38_4',
    [string]$Build = 'build_cuda_cmake/stage2_d_search_0_39_0'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$basePath = (Resolve-Path -LiteralPath $BaseBuild).Path
$buildPath = [IO.Path]::GetFullPath((Join-Path $repo $Build))
if (-not $buildPath.StartsWith($repo + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The development build must stay inside the repository'
}
if (Test-Path -LiteralPath $buildPath) { throw 'Use a fresh development build directory' }
$manifest = Get-Content -LiteralPath (Join-Path $basePath 'build_manifest.json') -Raw | ConvertFrom-Json
$objects = Join-Path $buildPath '_objects'
New-Item -ItemType Directory -Path $objects -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $basePath 'build_manifest.json') -Destination $buildPath
Copy-Item -LiteralPath (Join-Path $basePath '_objects/ecm_cuda_stage2.obj') -Destination $objects
# The formal builder verifies every CUDA dependency and the object hash before
# reuse. No settings, production directories or old build files are modified.
& (Join-Path $repo 'tools/build/build_stage2_local.ps1') -Build $buildPath -Engine production `
    -Arch $manifest.architecture -GlBackend $manifest.gl_backend -AddSubMask $manifest.add_sub_mask `
    -OuterUnrollU $manifest.outer_unroll_u -SplitCompile $manifest.split_compile -HostOnly
if ($LASTEXITCODE) { throw "Stage2 host build failed: $LASTEXITCODE" }
