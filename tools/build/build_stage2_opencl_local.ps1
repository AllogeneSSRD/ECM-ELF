#Requires -Version 5.1
param([string]$Build='build_stage2_opencl_local',[string]$OpenClRoot='',
      [string]$Gmp='',[string]$VcVars='')
$ErrorActionPreference='Stop'
& (Join-Path $PSScriptRoot 'test/build_opencl_stage2_ntt.ps1') -Build $Build -OpenClRoot $OpenClRoot -Gmp $Gmp -VcVars $VcVars -WorkerOnly
