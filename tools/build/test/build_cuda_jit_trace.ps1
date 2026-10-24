#Requires -Version 5.1
param([string]$Build='build_cuda_jit_trace')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
$cuda=Resolve-EcmCudaToolkit '80' -Version '13.3';$vc=Resolve-EcmVcEnvironment $cuda
if(-not [IO.Path]::IsPathRooted($Build)){$Build=Join-Path $repo $Build}
New-Item -ItemType Directory -Force $Build | Out-Null
$cupti=Join-Path $cuda.Root 'extras/CUPTI'
$command=$vc.Setup+' && "'+$vc.Compiler+'" /nologo /std:c++17 /EHsc /MD /O2 /utf-8 /LD /I "'+$cupti+'/include" /I "'+$cuda.Root+'/include" "'+$repo+'/tools/test/cuda_jit_trace.cpp" /Fo"'+$Build+'/cuda_jit_trace.obj" /link /LIBPATH:"'+$cupti+'/lib64" cupti.lib /IMPLIB:"'+$Build+'/cuda_jit_trace.lib" /OUT:"'+$Build+'/cuda_jit_trace.dll"'
& cmd.exe /d /c $command
if($LASTEXITCODE){throw 'JIT trace injection build failed'}
Get-ChildItem -LiteralPath (Join-Path $cupti 'lib64') -Filter 'cupti64*.dll' | Copy-Item -Destination $Build -Force
Write-Host "JIT observer: $Build/cuda_jit_trace.dll"
