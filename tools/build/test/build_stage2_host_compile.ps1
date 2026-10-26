# Compile real application/task/process/input implementations without CUDA SDK.
# Actual device policy and planning memory queries retain their runtime dependency.
param([Parameter(Mandatory=$true)][string]$WorkerBuild,
      [Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$worker=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkerBuild)
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(Test-Path -LiteralPath $output){throw 'Use a fresh output directory'}
$m=Get-Content -LiteralPath (Join-Path $worker 'build_manifest.json') -Raw -Encoding UTF8|ConvertFrom-Json
if(-not $m.host_compiler -or -not $m.compile_setup){throw 'Formal host toolchain identity required'}
. (Join-Path $PSScriptRoot '../internal/cuda_build_state.ps1')
$inventory=Get-EcmCudaInventory $repo
$sources=@('src/ecm/stage2/cuda/main.cpp',
    'src/ecm/application/configuration/stage2_identity.cpp',
    'src/ecm/stage2/task.cpp','src/ecm/stage2/input.cpp',
    'src/ecm/stage2/process.cpp')
foreach($source in $sources){if($source -notin $inventory.groups.STAGE2_HOST){throw "Missing production source: $source"}}
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $output
$commands=@();$hashes=[ordered]@{};$objects=[ordered]@{}
foreach($source in $sources){
    $object=Join-Path $output ($source.Replace('/','_')+'.obj');$objects[$source]=$object
    $commands+=('"'+$m.host_compiler+'" /nologo /TP /std:c++17 /O2 /EHsc /MT /utf-8 /wd4819 /DNOMINMAX'+
        ' /I "'+$repo+'/include" /I "'+$repo+'/src" /I "'+$m.gmp_root+'/include"'+
        ' /c "'+$source+'" /Fo"'+$object+'" /sourceDependencies "'+$object+'.deps.json"')
    $commands+='if errorlevel 1 exit /b 1'
    $hashes[$source]=(Get-FileHash -LiteralPath (Join-Path $repo $source)).Hash
}
$batch=Join-Path $output 'build.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($m.compile_setup)`r`nif errorlevel 1 exit /b 1`r`n"+($commands -join "`r`n")+"`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Out-File -LiteralPath (Join-Path $output 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $output 'build.log') -Tail 35;throw 'Stage2 host compile failed'}
$headers=[ordered]@{};$objectHashes=[ordered]@{}
foreach($source in $sources){
    $object=$objects[$source];$deps=Get-Content -LiteralPath "$object.deps.json" -Raw -Encoding UTF8|ConvertFrom-Json
    foreach($path in $deps.Data.Includes){
        if($path.Replace('\','/').StartsWith($m.cuda_root.Replace('\','/'),[StringComparison]::OrdinalIgnoreCase)){
            throw "Host implementation unexpectedly depends on CUDA SDK: $path"
        }
        $headers[$path]=(Get-FileHash -LiteralPath $path).Hash
    }
    $objectHashes[$source]=(Get-FileHash -LiteralPath $object).Hash
}
[ordered]@{complete=$true;compile_only=$true;compiler=$m.host_compiler;crt='MT';cuda_compiler_invocations=0;
    cuda_sdk_include_paths=0;cuda_sdk_headers=0;source_hashes=$hashes;header_hashes=$headers;objects=$objectHashes}|
    ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
Write-Host 'Compiled five actual Stage2 host implementations without CUDA SDK headers.'
