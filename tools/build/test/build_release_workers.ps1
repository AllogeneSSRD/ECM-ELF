#Requires -Version 5.1
param([ValidateSet('cu11.8','cu13.3')][string]$Profile='cu13.3',
    [ValidateSet('stage1','stage2','both')][string]$Stage='both',
    [string]$BuildRoot='build_release_test',[string]$Tiers='', [int]$Jobs=8,
    [string]$Gmp='', [string]$CudaRoot='', [string]$VcVars='', [switch]$SkipBuild)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
. (Join-Path $PSScriptRoot '../internal/release_profiles.ps1')
$target=Get-EcmReleaseTarget $Profile
$tools=Resolve-EcmReleaseToolchain $target $CudaRoot $VcVars
. (Join-Path $PSScriptRoot '../internal/gmp_dependency.ps1')
$dependency=Resolve-EcmGmp $repo $Gmp -Release
if(-not [IO.Path]::IsPathRooted($BuildRoot)){$BuildRoot=Join-Path $repo $BuildRoot}
$records=Join-Path $BuildRoot 'components'
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $BuildRoot
Initialize-EcmBuildDirectory $records
if($Stage -in @('stage1','both')) {
    $arguments=@('-BuildRoot',$BuildRoot,'-OutDir',(Join-Path $repo 'release/.stage1-component'),
        '-Archs',$target.Sass,'-PtxArch',$target.Ptx,'-Jobs',$Jobs,
        '-CudaRoot',$tools.Cuda.Root,'-VcVars',$tools.Vc.VcVars,'-Gmp',$dependency.Root,'-SkipSplit',
        '-ResultFile',(Join-Path $records "$Profile.stage1.json"))
    if($SkipBuild){$arguments+='-SkipBuild'}
    if($Tiers){$arguments+=@('-Tiers',$Tiers)}
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot '../internal/build_stage1_release_group.ps1') @arguments
    if($LASTEXITCODE){throw 'Release Stage1 build failed'}
}
if($Stage -in @('stage2','both')) {
    if($SkipBuild){throw 'Use the complete release identity preflight for Stage2 SkipBuild'}
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot '../build_stage2_local.ps1') `
        -BuildRoot $BuildRoot -BuildRole release -Arch ('sm_'+$target.Sass) -PtxArch $target.Ptx `
        -SplitCompile $target.SplitCompile -Jobs $Jobs -CudaRoot $tools.Cuda.Root -VcVars $tools.Vc.VcVars -Gmp $dependency.Root `
        -ResultFile (Join-Path $records "$Profile.stage2.json")
    if($LASTEXITCODE){throw 'Release Stage2 build failed'}
}
