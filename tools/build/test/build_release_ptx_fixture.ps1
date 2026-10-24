# Test-only PTX image derivation. Never modifies release objects or executables.
param([string]$BuildRoot='build_cuda_release',[Parameter(Mandatory=$true)][string]$Output,
    [ValidateSet('stage1','stage2','both')][string]$Stage='both')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
& python (Join-Path $repo 'tools/test/prepare_ptx_fixtures.py') --build-root $BuildRoot --output $Output --stage $Stage
if($LASTEXITCODE){throw 'PTX-only test fixture derivation failed'}
