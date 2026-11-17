#Requires -Version 5.1
param([string]$Build='build_stage2_cpu_local',[int]$Jobs=8,[string]$VcVars='')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if(-not [IO.Path]::IsPathRooted($Build)){$Build=Join-Path $repo $Build}
. (Join-Path $PSScriptRoot 'internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
& (Join-Path $PSScriptRoot 'test/build_cpu_stage2.ps1') -Build $Build -Jobs $Jobs -VcVars $VcVars -WithGwDll
foreach($file in @('ecm_stage2.exe','gwnum.dll','gmp-10.dll')){
    Copy-Item -LiteralPath (Join-Path $Build "native/$file") -Destination $Build -Force
}
Copy-Item (Join-Path $Build 'native/COPYING*') -Destination $Build -Force
& (Join-Path $Build 'ecm_stage2.exe') --version
if($LASTEXITCODE){throw 'Local CPU Stage2 version probe failed'}
