#Requires -Version 5.1
param([string]$Build='build_dev/build_cpu_stage2',[int]$Jobs=8,[string]$VcVars='',[switch]$GwPoly,[switch]$WithGwDll)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
if(-not $VcVars){
    $VcVars=Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat' |
        Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if(-not $VcVars -or -not (Test-Path -LiteralPath $VcVars)){throw 'MSVC x64 environment not found'}
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
$project=Join-Path $Build 'project';New-Item -ItemType Directory -Force $project | Out-Null
$gmpRoot=(Join-Path $repo 'third_party/gmp-zen3/dist').Replace('\','/')
$repoUnix=$repo.Replace('\','/')
$cmake=@"
cmake_minimum_required(VERSION 3.20)
project(ecm_cpu_stage2_build C CXX)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
set(GMP_INCLUDE_DIR "$gmpRoot/include")
set(GMP_LIBRARY "$gmpRoot/lib/gmp.lib")
include("$repoUnix/tools/build/internal/ecm_cpu_stage2.cmake")
"@
if($WithGwDll){
    if($GwPoly){throw 'Choose either the local DLL or the static comparison build'}
    $cmake+=@"

set(ECM_GW_STAGE2 ON CACHE BOOL "" FORCE)
add_subdirectory("$repoUnix/tools/build/gwnum" gwnum)
set_target_properties(ecm_gwnum PROPERTIES LIBRARY_OUTPUT_DIRECTORY "$Build/native" RUNTIME_OUTPUT_DIRECTORY "$Build/native")
"@.Replace('\','/')
}
[IO.File]::WriteAllText((Join-Path $project 'CMakeLists.txt'),$cmake,[Text.UTF8Encoding]::new($false))
$gwOption=if($GwPoly){'ON'}else{'OFF'}
$commands=@(('cmake -S "{0}" -B "{1}/native" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DECM_CPU_GW_POLY={2}' -f $project,$Build,$gwOption),
            ('cmake --build "{0}/native" --parallel {1}' -f $Build,$Jobs))
$batch=Join-Path $Build 'build.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`ncall ""$VcVars"" >nul 2>&1`r`nif errorlevel 1 exit /b 1`r`n"+($commands -join "`r`nif errorlevel 1 exit /b 1`r`n")+"`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1 | Out-File (Join-Path $Build 'build.log') -Encoding utf8
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 50;throw 'CPU Stage2 build failed'}
Copy-Item "$gmpRoot/bin/gmp-10.dll" "$Build/native" -Force
Copy-Item "$repo/.refactor/gmp-ecm/COPYING*" "$Build/native" -Force
& "$Build/native/ecm_stage2.exe" --help
if($LASTEXITCODE -ne 0){throw 'CPU Stage2 launch failed'}
