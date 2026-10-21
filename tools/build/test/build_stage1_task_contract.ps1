# Compile the real host task module and file collaborators; no CUDA compiler/SDK.
param([Parameter(Mandatory=$true)][string]$WorkerBuild,
      [Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$worker=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkerBuild)
$buildPath=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(Test-Path -LiteralPath $buildPath){throw 'Use a fresh build directory'}
$manifest=Get-Content -LiteralPath (Join-Path $worker 'stage1_local_manifest.json') -Raw|ConvertFrom-Json
if(-not $manifest.host_compiler -or -not $manifest.compile_setup){throw 'Formal host toolchain identity required'}
. (Join-Path $PSScriptRoot '../internal/cuda_build_state.ps1')
$inventory=Get-EcmCudaInventory $repo
$production=@($inventory.groups.STAGE1_COMMON)+@($inventory.groups.STAGE1_CUDA)
$sources=@('src/ecm/service/stage1/task.cpp','src/ecm/storage/checkpoint/ecm_checkpoint.cpp',
    'src/ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.cpp','src/ecm/platform/logging.cpp','src/ecm/application/configuration/runtime.cpp',
    'src/ecm/core/stage1/ecm_stage1_exp.cpp','src/ecm/platform/stage1_progress.cpp')
foreach($source in $sources){if($source -notin $production){throw "Missing production collaborator: $source"}}
$sources+=@('tools/test/stage1_task_contract.cpp')
New-Item -ItemType Directory -Path $buildPath|Out-Null
$objects=@();$commands=@();$hashes=[ordered]@{}
foreach($source in $sources){
    $name=$source.Replace('/','_')+'.obj';$object=Join-Path $buildPath $name;$objects+=('"'+$object+'"')
    $commands+=('"'+$manifest.host_compiler+'" /nologo /TP /std:c++17 /O2 /EHsc /MD /utf-8 /wd4819'+
        ' /I "'+$repo+'/include" /I "'+$repo+'/src" /I "'+$repo+'/src" /I "'+$manifest.gmp_root+'/include"'+
        ' /c "'+$source+'" /Fo"'+$object+'" /sourceDependencies "'+$object+'.deps.json"')
    $commands+='if errorlevel 1 exit /b 1'
    $hashes[$source]=(Get-FileHash -LiteralPath (Join-Path $repo $source)).Hash
}
$exe=Join-Path $buildPath 'stage1_task_contract.exe'
$commands+=('"'+$manifest.host_compiler+'" /nologo '+($objects -join ' ')+' "'+$manifest.gmp_root+'/lib/gmp.lib" /Fe"'+$exe+'"')
$batch=Join-Path $buildPath 'build.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($manifest.compile_setup)`r`nif errorlevel 1 exit /b 1`r`n"+($commands -join "`r`n")+"`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Out-File -LiteralPath (Join-Path $buildPath 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $buildPath 'build.log') -Tail 35;throw 'Task host contract build failed'}
Copy-Item -LiteralPath (Join-Path $manifest.gmp_root 'bin/gmp-10.dll') -Destination $buildPath
$headers=[ordered]@{}
foreach($source in $sources){
    $object=Join-Path $buildPath ($source.Replace('/','_')+'.obj')
    $deps=Get-Content -LiteralPath "$object.deps.json" -Raw -Encoding UTF8|ConvertFrom-Json
    foreach($path in $deps.Data.Includes){
        if($path.Replace('\','/').StartsWith($manifest.cuda_root.Replace('\','/'),[StringComparison]::OrdinalIgnoreCase)){
            throw "Host task test unexpectedly depends on CUDA SDK: $path"
        }
        $headers[$path]=(Get-FileHash -LiteralPath $path).Hash
    }
}
[ordered]@{complete=$true;compiler=$manifest.host_compiler;crt='MD';cuda_compiler_invocations=0;cuda_sdk_include_paths=0;
    cuda_sdk_headers=0;source_hashes=$hashes;header_hashes=$headers;exe_sha256=(Get-FileHash -LiteralPath $exe).Hash}|
    ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $buildPath 'manifest.json') -Encoding utf8
Write-Host "Built real host task contract: $exe"
