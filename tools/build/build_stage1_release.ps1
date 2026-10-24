#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('all','cu11.8','cu13.3')][string]$Profile='all',
    [string]$BuildRoot='build_cuda_release', [string]$OutDir='release',
    [ValidateRange(1,64)][int]$Jobs=8,
    [string]$Gmp='', [string]$Cuda118Root='', [string]$Cuda133Root='', [string]$VcVars='',
    [switch]$SkipBuild
)
$parameters=@{Profile=$Profile;Component='stage1';BuildRoot=$BuildRoot;OutDir=$OutDir;Jobs=$Jobs;SkipBuild=$SkipBuild;SkipPackage=$true}
foreach($name in @('Gmp','Cuda118Root','Cuda133Root','VcVars')) {if(Get-Variable $name -ValueOnly){$parameters[$name]=Get-Variable $name -ValueOnly}}
& (Join-Path $PSScriptRoot 'build_release.ps1') @parameters
exit $LASTEXITCODE