# Compile the real host construction/result code plus the private CUDA support
# seam. The latter substitutes deallocation and performs no GPU operation.
param([Parameter(Mandatory=$true)][string]$WorkerBuild,
      [Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$worker=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkerBuild)
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(Test-Path -LiteralPath $output){throw 'Use a fresh output directory'}
$m=Get-Content -LiteralPath (Join-Path $worker 'stage1_local_manifest.json') -Raw|ConvertFrom-Json
if(-not $m.compile_setup -or -not $m.host_compiler -or -not $m.cuda_root){throw 'Formal local toolchain manifest required'}
New-Item -ItemType Directory -Path $output|Out-Null
$sources=@('src/ecm/backend/cuda/stage1/montgomery/host_math.cpp',
    'src/ecm/backend/cuda/stage1/edwards/runtime.cpp','src/ecm/core/stage1/ecm_stage1_exp.cpp',
    'src/ecm/service/stage1/execution_contract.cpp','tools/test/stage1_cuda_host_contract.cpp')
$commands=@();$objects=@();$hashes=[ordered]@{}
foreach($source in $sources){
    $object=Join-Path $output ($source.Replace('/','_')+'.obj');$objects+=('"'+$object+'"')
    $commands+=('"'+$m.host_compiler+'" /nologo /TP /std:c++17 /O2 /EHsc /MD /utf-8 /DECM_CUDA_FULL_BUILD=1'+
        ' /DECM_TIERS_RESTRICTED /DECM_TIER_384 /I "'+$repo+'/include" /I "'+$repo+'/src" /I "'+$m.gmp_root+'/include"'+
        ' /c "'+$source+'" /Fo"'+$object+'" /sourceDependencies "'+$object+'.deps.json"')
    $commands+='if errorlevel 1 exit /b 1'
    $hashes[$source]=(Get-FileHash -LiteralPath $source).Hash
}
$hostExe=Join-Path $output 'cuda_host_contract.exe'
$commands+=('"'+$m.host_compiler+'" /nologo '+($objects -join ' ')+' "'+$m.gmp_root+'/lib/gmp.lib" /Fe"'+$hostExe+'"')
$commands+='if errorlevel 1 exit /b 1'
$resourceExe=Join-Path $output 'cuda_resources.exe'
$commands+=('"'+$m.cuda_root+'/bin/nvcc.exe" -std=c++17 -arch=sm_'+$m.architecture+' -ccbin "'+$m.host_compiler+
    '" -I "'+$repo+'/include" -I "'+$repo+'/src" -I "'+$repo+'/third_party/cgbn/include" -I "'+$m.gmp_root+
    '/include" -Xcompiler=/MD,/EHsc,/utf-8 "tools/test/adapters/stage1_cuda_resources.cu" -L "'+$m.gmp_root+'/lib" -lgmp -o "'+$resourceExe+'"')
$commands+='if errorlevel 1 exit /b 1'
$script=Join-Path $output 'build.cmd'
[IO.File]::WriteAllText($script,"@echo off`r`n$($m.compile_setup)`r`nif errorlevel 1 exit /b 1`r`n"+($commands -join "`r`n")+"`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $script 2>&1|Out-File -LiteralPath (Join-Path $output 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $output 'build.log') -Tail 30;throw 'CUDA Stage1 contracts build failed'}
Copy-Item -LiteralPath (Join-Path $m.gmp_root 'bin/gmp-10.dll') -Destination $output
foreach($exe in @($hostExe,$resourceExe)){
    # CGBN diagnostics on stderr are expected in the propagation fixture.
    & cmd.exe /d /c ('"'+$exe+'" > "'+$exe+'.log" 2>&1')
    if($LASTEXITCODE){Get-Content -LiteralPath ($exe+'.log');throw 'CUDA Stage1 contract failed'}
}
$headers=[ordered]@{}
foreach($source in $sources){
    $object=Join-Path $output ($source.Replace('/','_')+'.obj')
    $deps=Get-Content -LiteralPath ($object+'.deps.json') -Raw|ConvertFrom-Json
    foreach($path in $deps.Data.Includes){
        if($path.Replace('\','/').StartsWith($m.cuda_root.Replace('\','/'),[StringComparison]::OrdinalIgnoreCase)){throw "Host construction includes SDK: $path"}
        $headers[$path]=(Get-FileHash -LiteralPath $path).Hash
    }
}
[ordered]@{complete=$true;gpu_runs=0;host_cuda_compiler_invocations=0;host_cuda_sdk_headers=0;
    resource_cuda_compiler_invocations=1;source_hashes=$hashes;host_header_hashes=$headers;
    host_binary_sha256=(Get-FileHash -LiteralPath $hostExe).Hash;
    resource_binary_sha256=(Get-FileHash -LiteralPath $resourceExe).Hash}|
    ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
Write-Host 'PASS real host construction and CUDA support contracts; gpu_runs=0'
