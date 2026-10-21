# Configure-only CXX fixture for the actual owner-graph collector. No CUDA/GPU.
param([Parameter(Mandatory=$true)][string]$WorkerBuild,
      [Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$worker=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkerBuild)
$out=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if(Test-Path -LiteralPath $out){throw 'Use a fresh output directory'}
$manifest=Get-Content -LiteralPath (Join-Path $worker 'stage1_local_manifest.json') -Raw|ConvertFrom-Json
if(-not $manifest.host_compiler -or -not $manifest.compile_setup){throw 'A formal Stage1 toolchain manifest is required'}
New-Item -ItemType Directory -Force (Join-Path $out 'source/src/ecm')|Out-Null
foreach($name in @('src/ecm/root.cpp','object.cpp','link.cpp','conditional.cpp')){
    [IO.File]::WriteAllText((Join-Path $out "source/$name"),'// configure-only fixture',[Text.UTF8Encoding]::new($false))
}
$fixture=@'
cmake_minimum_required(VERSION 3.20)
project(EcmOwnerGraph LANGUAGES CXX)
include("@HELPER@")
add_library(ecm INTERFACE)
add_library(real_objects OBJECT object.cpp)
add_library(real_link STATIC link.cpp)
add_library(conditioned STATIC conditional.cpp)
add_library(imported_iface INTERFACE IMPORTED)
add_library(ecm_cuda STATIC src/ecm/root.cpp $<TARGET_OBJECTS:real_objects>)
set_property(TARGET ecm_cuda APPEND PROPERTY INTERFACE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/ecm/root.cpp")
target_link_libraries(ecm_cuda PRIVATE real_link imported_iface $<$<CONFIG:Release>:conditioned>)
ecm_cuda_write_target_graph()
'@
$fixture=$fixture.Replace('@HELPER@',((Join-Path $repo 'tools/build/internal/cuda_sources.cmake').Replace('\','/')))
[IO.File]::WriteAllText((Join-Path $out 'source/CMakeLists.txt'),$fixture,[Text.UTF8Encoding]::new($false))
$command="cmake -S `"$out/source`" -B `"$out/build`" -G `"NMake Makefiles`" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=`"$($manifest.host_compiler.Replace('\','/'))`""
$batch=Join-Path $out 'configure.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($manifest.compile_setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Out-File -LiteralPath (Join-Path $out 'configure.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $out 'configure.log') -Tail 20;throw 'Owner graph fixture configure failed'}
$graph=Get-Content -LiteralPath (Join-Path $out 'build/cuda_target_graph.json') -Raw|ConvertFrom-Json
$expected=@('ecm_cuda','real_objects','real_link','conditioned')
$actual=@($graph.targets.ecm_cuda)
if(@(Compare-Object ($expected|Sort-Object) ($actual|Sort-Object)).Count){throw "Owner graph differs: $actual"}
[ordered]@{complete=$true;targets=$actual;checks=@('source directory collision excluded','explicit TARGET_OBJECTS retained','direct link retained','conditional link retained','imported target excluded');cuda_compiles=0;gpu_runs=0}|
    ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $out 'result.json') -Encoding utf8
Write-Host 'PASS: actual CMake owner graph distinguishes source paths and targets'
