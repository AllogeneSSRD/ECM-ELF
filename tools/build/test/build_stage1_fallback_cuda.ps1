#Requires -Version 5.1
param([string]$WorkerBuild='build_stage1_local')
$ErrorActionPreference='Stop'
& (Join-Path $PSScriptRoot '../build_stage1_local.ps1') -BuildDir $WorkerBuild -HostOnly
if($LASTEXITCODE){throw 'Stage1 CUDA fallback host build failed'}
