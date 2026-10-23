#Requires -Version 5.1
param([string]$Build='build_go_fallback_stage1')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
# Compile the real Stage1/OpenCL frontend and exercise its CPU route; no CUDA TUs.
& cmake -S $repo -B $Build -DECM_ENABLE_CUDA=OFF -DOPENSSL_ROOT_DIR=D:/code/vcpkg/installed/x64-windows
if($LASTEXITCODE){throw 'Stage1 fallback test configure failed'}
& cmake --build $Build --config Release --target ecm -- /m:1 /nr:false
if($LASTEXITCODE){throw 'Stage1 fallback test build failed'}
