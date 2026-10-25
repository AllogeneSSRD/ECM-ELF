#Requires -Version 5.1
# Build a frozen pre-NAF source snapshot, including the native M8191 Mont tier.
param([Parameter(Mandatory=$true)][string]$Source,
      [string]$Build='build_cuda_cmake/edwards_initial_native',
      [string]$Tiers='384,768,1024,2560,4096,9216')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$Source=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Source)
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(-not (Test-Path -LiteralPath (Join-Path $Source 'CMakeLists.txt'))){throw 'Missing frozen CMake source'}
if(Test-Path -LiteralPath $Build){throw 'Use a fresh baseline build directory'}
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
& (Join-Path $Source 'tools/build/build_stage1_local.ps1') -BuildDir $Build -Arch 89 -Jobs 8 -Tiers $Tiers `
    -Gmp (Join-Path $repo 'third_party/gmp-zen3/dist') -Extra '-DECM_BUILD_TOOLS=OFF -DECM_BUILD_GUI=OFF'
if($LASTEXITCODE -ne 0){throw 'Frozen baseline build failed'}
