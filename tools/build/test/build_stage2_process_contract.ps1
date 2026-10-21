# Compile/run actual Stage2 curve-process adapter without a CUDA SDK.
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
$sources=@('src/ecm/platform/process/stage2_curve_process.cpp')
foreach($source in $sources){if($source -notin $inventory.groups.STAGE2_HOST){throw "Missing production collaborator: $source"}}
$sources+=@('tools/test/stage2_process_contract.cpp')
New-Item -ItemType Directory -Path $output|Out-Null
$commands=@();$objects=@();$hashes=[ordered]@{}
foreach($source in $sources){
    $object=Join-Path $output ($source.Replace('/','_')+'.obj');$objects+=('"'+$object+'"')
    $commands+=('"'+$m.host_compiler+'" /nologo /TP /std:c++17 /O2 /EHsc /MT /utf-8 /wd4819 /DNOMINMAX'+
        ' /I "'+$repo+'/include" /I "'+$repo+'/src" /c "'+$source+'" /Fo"'+$object+'" /sourceDependencies "'+$object+'.deps.json"')
    $commands+='if errorlevel 1 exit /b 1'
    $hashes[$source]=(Get-FileHash -LiteralPath (Join-Path $repo $source)).Hash
}
$exe=Join-Path $output 'stage2_process_contract.exe'
$commands+=('"'+$m.host_compiler+'" /nologo '+($objects -join ' ')+' /Fe"'+$exe+'"')
$batch=Join-Path $output 'build.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($m.compile_setup)`r`nif errorlevel 1 exit /b 1`r`n"+($commands -join "`r`n")+"`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Out-File -LiteralPath (Join-Path $output 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $output 'build.log') -Tail 35;throw 'Process contract compile failed'}
$headers=[ordered]@{}
foreach($source in $sources){
    $object=Join-Path $output ($source.Replace('/','_')+'.obj')
    $deps=Get-Content -LiteralPath "$object.deps.json" -Raw -Encoding UTF8|ConvertFrom-Json
    foreach($path in $deps.Data.Includes){
        if($path.Replace('\','/').StartsWith($m.cuda_root.Replace('\','/'),[StringComparison]::OrdinalIgnoreCase)){
            throw "Process contract unexpectedly depends on CUDA SDK: $path"
        }
        $headers[$path]=(Get-FileHash -LiteralPath $path).Hash
    }
}
$run=Join-Path $output 'run'
$runLog=Join-Path $output 'run.log'
$runBatch=Join-Path $output 'run.cmd'
[IO.File]::WriteAllText($runBatch,"@echo off`r`n`"$exe`" `"$run`" >`"$runLog`" 2>&1`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $runBatch
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $output 'run.log');throw 'Process contract failed'}
$report=Get-Content -LiteralPath (Join-Path $output 'run.log') -Raw -Encoding UTF8
$result=@($report -split "`r?`n"|Where-Object {$_ -match '^\{"passed":'})[-1]|ConvertFrom-Json
if(-not $result.passed -or $result.cuda_devices -ne 0 -or $result.engine_links -ne 0){throw 'Missing fixture completion'}
[ordered]@{complete=$true;compiler=$m.host_compiler;crt='MT';cuda_compiler_invocations=0;cuda_sdk_include_paths=0;
    cuda_sdk_headers=0;source_hashes=$hashes;header_hashes=$headers;processes=$result.processes;wire_checks=$result.wire_checks;engine_links=0;exe_sha256=(Get-FileHash -LiteralPath $exe).Hash}|
    ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
Write-Host "Stage2 process checks passed: $($result.wire_checks); CUDA SDK headers=0"
