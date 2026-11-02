# Exact shared CGBN square and existing Montgomery reduction, isolated from workers.
param([string]$Build='build_dev/cgbn_square_probe',[string]$Arch='sm_89',
      [ValidateRange(128,32768)][int]$Bits=8192,[ValidateSet(4,8,16,32)][int]$Tpi=16,
      [string]$Modes='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23',
      [ValidateRange(1,64)][int]$SplitCompile=8,[switch]$Rebuild)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
if($Bits%32){throw 'Bits must be divisible by 32'}
$modeMask=0
foreach($mode in $Modes.Split(',')) {
    if($mode -notmatch '^(?:[0-9]|1[0-9]|2[0-3])$'){throw 'Modes must be comma-separated integers in [0,23]'}
    $modeMask=$modeMask -bor (1 -shl [int]$mode)
}
. (Join-Path $PSScriptRoot '../internal/cuda_experiment_probe.ps1')
Invoke-EcmExperimentBuild -Repo $repo -Build $Build -Arch $Arch `
    -Sources @('tools/test/cgbn_square_probe.cu') -Name 'cgbn_square_probe' `
    -Defines @("ECM_SQUARE_BITS=$Bits","ECM_SQUARE_TPI=$Tpi","ECM_SQUARE_MODE_MASK=$modeMask") -SplitCompile $SplitCompile -Rebuild:$Rebuild
