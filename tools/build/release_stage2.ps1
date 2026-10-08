#Requires -Version 5.1
<#
.SYNOPSIS
Build and package the standalone CUDA Stage2 release candidate for one architecture.
.DESCRIPTION
Builds the production engine with fixed arithmetic defaults, then includes the
GMP runtime, shared INI template, empty Stage2 queue and user documentation.
Existing ecm.ini and user queue files are never overwritten. An optional
previously qualified Stage1 executable can be included with -Stage1Exe.
This script does not run tests or certify other GPU architectures.
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/release_stage2.ps1 -Arch sm_89 -SplitCompile 6
#>
param(
    [ValidatePattern('^sm_[0-9]+$')][string]$Arch = 'sm_89',
    [string]$Build = 'build_cuda_cmake/release_ux_sm89',
    [string]$OutDir = 'dist/cuda-stage2-sm89',
    [ValidateRange(1,64)][int]$SplitCompile = 6,
    [string]$Stage1Exe = '',
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
& cmake "-DECM_CONFIG_ROOT=$repo" -P (Join-Path $PSScriptRoot 'check_ecm_config.cmake')
if ($LASTEXITCODE -ne 0) { throw 'Generated ECM config is stale; run python tools/gen/generate_ecm_config.py' }
function Resolve-RepoPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $repo $Value))
}
$Build = Resolve-RepoPath $Build
$OutDir = Resolve-RepoPath $OutDir
if ($Build.TrimEnd('\') -eq $OutDir.TrimEnd('\')) { throw 'Build and package directories must differ' }
if (-not $SkipBuild) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'build_ecm_cuda_stage2.ps1') `
        -Build $Build -Engine production -Arch $Arch -SplitCompile $SplitCompile
    if ($LASTEXITCODE -ne 0) { throw 'Stage2 release build failed' }
}
$manifestPath = Join-Path $Build 'build_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$exe = Join-Path $Build 'ecm_cuda_stage2.exe'
if ($manifest.engine -ne 'production' -or $manifest.architecture -ne $Arch -or
    $manifest.add_sub_mask -ne 1 -or $manifest.outer_unroll_u -ne 0 -or $manifest.gl_fixed_mode -ne 3 -or
    $manifest.sha256 -ne (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash) {
    throw 'Build manifest does not match the requested production candidate'
}
# SkipBuild is a packaging convenience, not permission to package stale sources.
foreach ($entry in $manifest.source_hashes.PSObject.Properties) {
    if ($entry.Value -ne (Get-FileHash -LiteralPath (Join-Path $repo $entry.Name) -Algorithm SHA256).Hash) {
        throw "Build sources changed: $($entry.Name); rebuild before packaging"
    }
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
Copy-Item -LiteralPath $exe -Destination $OutDir -Force
Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $OutDir 'stage2_build_manifest.json') -Force
Copy-Item -LiteralPath (Join-Path $Build 'gmp-10.dll') -Destination $OutDir -Force
Copy-Item -LiteralPath (Join-Path $repo 'config/ecm.ini.example') -Destination $OutDir -Force
$ini = Join-Path $OutDir 'ecm.ini'
if (-not (Test-Path -LiteralPath $ini)) { Copy-Item -LiteralPath (Join-Path $repo 'config/ecm.ini.example') -Destination $ini }
$queue = Join-Path $OutDir 'stage2_worktodo.txt'
if (-not (Test-Path -LiteralPath $queue)) {
    $queueHeader = '# Add ECMSTAGE2=k,b,n,c,filename[,B2-or-zero][,skip_curves][,num_curves][,"known-factors"].'
    $queueHeader += [Environment]::NewLine
    [IO.File]::WriteAllText($queue, $queueHeader, (New-Object Text.UTF8Encoding $false))
}
Copy-Item -LiteralPath (Join-Path $repo 'docs/ECM_CUDA_STAGE2_RELEASE.md') -Destination (Join-Path $OutDir 'README.md') -Force
Copy-Item -LiteralPath (Join-Path $repo 'docs/DEV_ECM_INI.md') -Destination $OutDir -Force
Copy-Item -LiteralPath (Join-Path $repo 'docs/ECM_INI_REFERENCE.md') -Destination $OutDir -Force
Copy-Item -LiteralPath (Join-Path $repo 'docs/DEV_ECM_CONFIG_SCHEMA.md') -Destination $OutDir -Force
if ($Stage1Exe) {
    $Stage1Exe = Resolve-RepoPath $Stage1Exe
    if (-not (Test-Path -LiteralPath $Stage1Exe)) { throw 'Qualified Stage1 executable not found' }
    Copy-Item -LiteralPath $Stage1Exe -Destination (Join-Path $OutDir 'ecm_cuda.exe') -Force
}
$hashes = [ordered]@{}
foreach ($name in @('ecm_cuda_stage2.exe','gmp-10.dll','ecm.ini.example','README.md','DEV_ECM_INI.md','ECM_INI_REFERENCE.md','DEV_ECM_CONFIG_SCHEMA.md','stage2_build_manifest.json')) {
    $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $OutDir $name) -Algorithm SHA256).Hash
}
if ($Stage1Exe) { $hashes['ecm_cuda.exe'] = (Get-FileHash -LiteralPath $Stage1Exe -Algorithm SHA256).Hash }
[ordered]@{
    architecture = $Arch
    status = 'release_candidate_build_only'
    stage1_included = [bool]$Stage1Exe
    stage1_rebuilt = $false
    tests_run = $false
    auto_b2_calibrated = $false
    hashes = $hashes
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutDir 'package_manifest.json') -Encoding UTF8
Write-Host "Packaged $Arch candidate: $OutDir"
