#Requires -Version 5.1
param(
    [Parameter(Mandatory=$true)][string]$BuildDir,
    [Parameter(Mandatory=$true)][string]$OutDir,
    [Parameter(Mandatory=$true)][string]$Archs,
    [Parameter(Mandatory=$true)][string]$Gmp
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
. (Join-Path $PSScriptRoot 'gmp_dependency.ps1')
$gmpDependency = Resolve-EcmGmp -Repo $repo -Gmp $Gmp -Release
$Gmp = $gmpDependency.Root
. (Join-Path $PSScriptRoot 'release_payload.ps1')
$splitPath = Join-Path $BuildDir 'split_manifest.json'
$split = Get-Content -LiteralPath $splitPath -Raw | ConvertFrom-Json
$manifestPath = Join-Path $BuildDir 'stage1_build_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($name in $gmpDependency.Hashes.Keys) {
    if ($manifest.dependency_hashes.$name -ne $gmpDependency.Hashes[$name]) { throw "Stage1 build/release GMP mismatch: $name" }
}
if ($split.build_manifest_sha256 -ne (Get-FileHash -LiteralPath $manifestPath).Hash) { throw 'Stage1 split/build manifest mismatch' }
Remove-EcmReleaseMetadata $OutDir
foreach ($arch in $Archs.Split(',')) {
    $name = 'ecm_cuda_sm' + $arch + '.exe'
    $exe = Join-Path $OutDir $name
    if ($split.executables.$name -ne (Get-FileHash -LiteralPath $exe).Hash) { throw "Stage1 split executable changed: $name" }
    $package = Join-Path $OutDir ('cuda-stage1-sm' + $arch)
    New-Item -ItemType Directory -Path $package -Force | Out-Null
    Remove-EcmReleaseMetadata $package
    Copy-Item -LiteralPath $exe -Destination (Join-Path $package 'ecm_cuda.exe') -Force
    Copy-Item -LiteralPath (Join-Path $Gmp 'bin/gmp-10.dll') -Destination $package -Force
    Copy-Item -LiteralPath (Join-Path $repo 'config/ecm.ini.example') -Destination $package -Force
    Copy-Item -LiteralPath (Join-Path $repo 'docs/ECM_INI_REFERENCE.md') -Destination $package -Force
    Copy-Item -LiteralPath (Join-Path $repo 'LICENSE') -Destination $package -Force
    Copy-Item -LiteralPath $gmpDependency.Copyright -Destination (Join-Path $package 'GMP-COPYRIGHT.txt') -Force
    $ini = Join-Path $package 'ecm.ini'
    if (-not (Test-Path -LiteralPath $ini)) { Copy-Item -LiteralPath (Join-Path $package 'ecm.ini.example') -Destination $ini }
    $queue = Join-Path $package 'worktodo.txt'
    if (-not (Test-Path -LiteralPath $queue)) {
        [IO.File]::WriteAllText($queue, '# Add Stage1 ECM worktodo assignments here.' + [Environment]::NewLine, (New-Object Text.UTF8Encoding $false))
    }
    $files = @('ecm_cuda.exe','gmp-10.dll','ecm.ini.example','LICENSE','GMP-COPYRIGHT.txt','ECM_INI_REFERENCE.md')
    $hashes = [ordered]@{}
    foreach ($file in $files) { $hashes[$file] = (Get-FileHash -LiteralPath (Join-Path $package $file)).Hash }
    [ordered]@{ architecture = "sm_$arch"; status = 'release_candidate_build_only'; tests_run = $false; hashes = $hashes } |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $BuildDir ('package_sm' + $arch + '_manifest.json')) -Encoding UTF8
    & (Join-Path $PSScriptRoot 'package_archive.ps1') -PackageDir $package -BuildDir $BuildDir -Stage stage1 -Files $files
    Write-Host "Packaged Stage1 sm_$arch : $package"
}
