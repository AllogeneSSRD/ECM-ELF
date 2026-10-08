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
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/build_stage2_release.ps1 -Arch sm_89 -SplitCompile 8
#>
param(
    [ValidatePattern('^sm_[0-9]+$')][string]$Arch = 'sm_89',
    [string]$Build = 'build_cuda_cmake/release_ux_sm89',
    [string]$OutDir = 'dist/cuda-stage2-sm89',
    [ValidateRange(1,64)][int]$SplitCompile = 8,
    [string]$Gmp = '',
    [string]$Stage1Exe = '',
    [string]$CudaRoot = '',
    [string]$VcVars = '',
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
. (Join-Path $PSScriptRoot 'gmp_dependency.ps1')
$gmpDependency = Resolve-EcmGmp -Repo $repo -Gmp $Gmp -Release
$Gmp = $gmpDependency.Root
. (Join-Path $PSScriptRoot 'cuda_toolchain.ps1')
$cuda = Resolve-EcmCudaToolkit -Archs $Arch -CudaRoot $CudaRoot
$vc = Resolve-EcmVcEnvironment $cuda -VcVars $VcVars
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
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot '../build_stage2_local.ps1') `
        -Build $Build -Engine production -Arch $Arch -SplitCompile $SplitCompile -Gmp $Gmp -CudaRoot $cuda.Root -VcVars $vc.VcVars
    if ($LASTEXITCODE -ne 0) { throw 'Stage2 release build failed' }
}
$manifestPath = Join-Path $Build 'build_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$exe = Join-Path $Build 'ecm_cuda_stage2.exe'
if ($manifest.engine -ne 'production' -or $manifest.architecture -ne $Arch -or
    $manifest.cuda_root -ne $cuda.Root -or $manifest.toolkit -ne $cuda.VersionText -or
    $manifest.host_compiler -ne $vc.Compiler -or $manifest.host_version -ne $vc.Version -or
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
if (-not $manifest.gmp_hashes) { throw 'Stage2 build does not record GMP identity; rebuild with the release GMP prefix' }
foreach ($name in $gmpDependency.Hashes.Keys) {
    if ($manifest.gmp_hashes.$name -ne $gmpDependency.Hashes[$name]) { throw "Stage2 build/release GMP mismatch: $name" }
}
if ((Get-FileHash -LiteralPath (Join-Path $Build 'gmp-10.dll')).Hash -ne $gmpDependency.Hashes['bin/gmp-10.dll']) {
    throw 'Stage2 build runtime DLL changed'
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
. (Join-Path $PSScriptRoot 'release_payload.ps1')
Remove-EcmReleaseMetadata $OutDir
Copy-Item -LiteralPath $exe -Destination $OutDir -Force
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
Copy-Item -LiteralPath (Join-Path $repo 'docs/ECM_INI_REFERENCE.md') -Destination $OutDir -Force
Copy-Item -LiteralPath (Join-Path $repo 'LICENSE') -Destination $OutDir -Force
Copy-Item -LiteralPath $gmpDependency.Copyright -Destination (Join-Path $OutDir 'GMP-COPYRIGHT.txt') -Force
if ($Stage1Exe) {
    $Stage1Exe = Resolve-RepoPath $Stage1Exe
    if (-not (Test-Path -LiteralPath $Stage1Exe)) { throw 'Qualified Stage1 executable not found' }
    Copy-Item -LiteralPath $Stage1Exe -Destination (Join-Path $OutDir 'ecm_cuda.exe') -Force
}
$hashes = [ordered]@{}
$files = @('ecm_cuda_stage2.exe','gmp-10.dll','ecm.ini.example','LICENSE','GMP-COPYRIGHT.txt','ECM_INI_REFERENCE.md')
foreach ($name in $files) {
    $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $OutDir $name) -Algorithm SHA256).Hash
}
if ($Stage1Exe) { $hashes['ecm_cuda.exe'] = (Get-FileHash -LiteralPath $Stage1Exe -Algorithm SHA256).Hash; $files += 'ecm_cuda.exe' }
[ordered]@{
    architecture = $Arch
    status = 'release_candidate_build_only'
    stage1_included = [bool]$Stage1Exe
    stage1_rebuilt = $false
    tests_run = $false
    auto_b2_calibrated = $false
    hashes = $hashes
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Build 'package_manifest.json') -Encoding UTF8
& (Join-Path $PSScriptRoot 'package_archive.ps1') -PackageDir $OutDir -BuildDir $Build -Stage stage2 -Files $files
Write-Host "Packaged $Arch candidate: $OutDir"
