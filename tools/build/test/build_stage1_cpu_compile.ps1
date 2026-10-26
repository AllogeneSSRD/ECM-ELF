# Compile the real CPU workflow and preparation with no CUDA compiler/SDK.
# Full behavior/link validation uses the production workers separately.
param([Parameter(Mandatory=$true)][string]$WorkerBuild,
      [Parameter(Mandatory=$true)][string]$Build,
      [switch]$IncludeGpuHost,
      [switch]$IncludeOpenCLHost,
      [switch]$IncludeMontgomeryHost,
      [switch]$EnablePracHost,
      [switch]$IncludeQueueHost)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$worker=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkerBuild)
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(Test-Path -LiteralPath $output){throw 'Use a fresh output directory'}
$m=Get-Content -LiteralPath (Join-Path $worker 'stage1_local_manifest.json') -Raw|ConvertFrom-Json
if(-not $m.host_compiler -or -not $m.compile_setup){throw 'Formal host toolchain identity required'}
New-Item -ItemType Directory -Path $output|Out-Null
[IO.File]::WriteAllText((Join-Path $output ".gitignore"),"*`n")
$sources=@('src/ecm/stage1/execution_contract.cpp','src/ecm/stage1/preparation.cpp','src/ecm/edwards_stage1/cpu/workflow.cpp','src/ecm/montgomery_stage1/cpu/workflow.cpp','src/ecm/stage1/cpu_support.cpp')
if($IncludeGpuHost){
    $sources+=@('src/ecm/stage1/execution.cpp',
        'src/ecm/montgomery_stage1/cuda/workflow.cpp',
        'src/ecm/edwards_stage1/cuda/workflow.cpp',
        'src/ecm/edwards_stage1/cuda/checkpoint.cpp',
        'src/ecm/edwards_stage1/cuda/host_math.cpp',
        'src/ecm/edwards_stage1/cuda/runtime.cpp')
}
if($IncludeMontgomeryHost){
    $sources+=@('src/ecm/montgomery_stage1/cuda/host_math.cpp',
        'src/ecm/montgomery_stage1/cuda/checkpoint.cpp',
        'src/ecm/montgomery_stage1/cuda/output.cpp',
        'src/ecm/montgomery_stage1/cuda/host_policy.cpp')
}
if($IncludeOpenCLHost){$sources+='src/ecm/montgomery_stage1/opencl/workflow.cpp'}
if($IncludeQueueHost){
    $sources+=@('src/io/normalize.cpp','src/ecm/application/cli/group_order.cpp',
        'src/io/cli.cpp','src/io/config.cpp','src/io/worktodo.cpp','src/io/files.cpp')
}
if($EnablePracHost -and -not $IncludeMontgomeryHost){throw 'EnablePracHost requires IncludeMontgomeryHost'}
$definitions=if($IncludeGpuHost){' /DECM_HAS_CUDA_STAGE1=1'}else{''}
$definitions+=' /DECM_CUDA_ENABLE_PRAC='+$(if($EnablePracHost){'1'}else{'0'})
$commands=@();$objects=[ordered]@{};$hashes=[ordered]@{}
foreach($source in $sources){
    $object=Join-Path $output ($source.Replace('/','_')+'.obj');$objects[$source]=$object
    $commands+=('"'+$m.host_compiler+'" /nologo /TP /std:c++17 /O2 /EHsc /MD /utf-8 /wd4819'+$definitions+
        ' /I "'+$repo+'/include" /I "'+$repo+'/src" /I "'+$m.gmp_root+'/include"'+
        ' /c "'+$source+'" /Fo"'+$object+'" /sourceDependencies "'+$object+'.deps.json"')
    $commands+='if errorlevel 1 exit /b 1'
    $hashes[$source]=(Get-FileHash -LiteralPath (Join-Path $repo $source)).Hash
}
$batch=Join-Path $output 'build.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($m.compile_setup)`r`nif errorlevel 1 exit /b 1`r`n"+($commands -join "`r`n")+"`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Out-File -LiteralPath (Join-Path $output 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $output 'build.log') -Tail 35;throw 'CPU workflow compile failed'}
$headers=[ordered]@{};$objectHashes=[ordered]@{}
foreach($source in $sources){
    $object=$objects[$source];$deps=Get-Content -LiteralPath "$object.deps.json" -Raw -Encoding UTF8|ConvertFrom-Json
    foreach($path in $deps.Data.Includes){
        if($path.Replace('\','/').StartsWith($m.cuda_root.Replace('\','/'),[StringComparison]::OrdinalIgnoreCase)){
            throw "CPU workflow unexpectedly depends on CUDA SDK: $path"
        }
        $headers[$path]=(Get-FileHash -LiteralPath $path).Hash
    }
    $objectHashes[$source]=(Get-FileHash -LiteralPath $object).Hash
}
[ordered]@{complete=$true;compile_only=$true;compiler=$m.host_compiler;crt='MD';cuda_compiler_invocations=0;
    gpu_host_workflow_included=[bool]$IncludeGpuHost;
    opencl_host_workflow_included=[bool]$IncludeOpenCLHost;
    montgomery_host_included=[bool]$IncludeMontgomeryHost;prac_host_enabled=[bool]$EnablePracHost;
    queue_host_included=[bool]$IncludeQueueHost;
    cuda_sdk_include_paths=0;cuda_sdk_headers=0;avx512_compile_option=$false;
    source_hashes=$hashes;header_hashes=$headers;objects=$objectHashes}|
    ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
Write-Host 'Compiled real host workflows without CUDA SDK headers.'
