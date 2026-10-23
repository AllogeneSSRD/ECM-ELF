#Requires -Version 5.1
param([string]$WorkerBuild='build_stage2_local')
$ErrorActionPreference='Stop'
# Reuse only CUDA objects accepted by the production dependency/profile guard.
# The formal host-only path refuses stale GPU objects instead of bypassing checks.
& (Join-Path $PSScriptRoot '../build_stage2_local.ps1') -Build $WorkerBuild -HostOnly
if($LASTEXITCODE){throw 'Stage2 fallback host build failed'}
