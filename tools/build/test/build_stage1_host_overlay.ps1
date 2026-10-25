# Isolated host refactor build. Verify the production CUDA source closure and
# object hashes, then compile the current host inventory. Explicitly opt in to
# rebuilding changed CUDA closures with the original compile commands and flags;
# all original objects and the production executable remain untouched.
param([string]$WorkerBuild='build_stage1_local',
      [Parameter(Mandatory=$true)][string]$Build,
      [switch]$RecompileChangedCuda)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$worker=(Resolve-Path -LiteralPath $WorkerBuild).Path
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
New-Item -ItemType Directory -Force -Path $output|Out-Null
[IO.File]::WriteAllText((Join-Path $output '.gitignore'),"*`n")
$manifestPath=Join-Path $worker 'stage1_local_manifest.json'
$m=Get-Content -LiteralPath $manifestPath -Raw|ConvertFrom-Json
if($m.engine -ne 'production' -or -not $m.objects -or -not $m.source_hashes){throw 'Production Stage1 source/object identity required'}
. (Join-Path $PSScriptRoot '../internal/cuda_build_state.ps1')
$inventory=Get-EcmCudaInventory $repo
$all=@($inventory.groups.STAGE1_COMMON)+@($inventory.groups.STAGE1_CUDA)
$cudaSources=@($all|Where-Object {$_ -match '\.cu$'})
$cudaEntries=@(Get-EcmTargetCompileEntries $worker ecm_cuda|Where-Object {$_.file -match '\.cu$'})
if($cudaSources.Count -ne $cudaEntries.Count){throw 'CUDA source inventory changed; use a normal CUDA build'}
$reused=[ordered]@{};$objects=@();$sourceHashes=[ordered]@{};$cudaCommands=@();$rebuilt=[ordered]@{}
foreach($entry in $cudaEntries) {
    $key=Get-EcmSourceKey $repo $entry.file
    if($key -notin $cudaSources){throw "Unexpected CUDA source: $key"}
    $inputs=Get-EcmCommandInputs $entry
    $closure=Get-EcmSourceClosure $repo @($entry.file) $inputs.IncludeDirs -Production
    $changed=$false
    foreach($source in $closure) {
        $hash=(Get-FileHash -LiteralPath (Join-Path $repo $source)).Hash
        $expected=$m.source_hashes.$source
        if(-not $expected -and (Join-Path $repo $source) -eq (Join-Path $m.gmp_root 'include/gmp.h')){$expected=$m.dependency_hashes.header}
        if($expected -ne $hash){
            if(-not $RecompileChangedCuda){throw "CUDA dependency changed; normal build required: $source"}
            $changed=$true
        }
        $sourceHashes[$source]=$hash
    }
    $object=Get-EcmObjectPath $entry $worker
    $relative=Get-EcmSourceKey $worker $object
    $hash=(Get-FileHash -LiteralPath $object).Hash
    if($m.objects.$relative -ne $hash){throw "CUDA object changed: $relative"}
    $reused[$object]=$hash # Verify the original objects remain untouched even when rebuilding.
    if($changed) {
        $fresh=Join-Path $output ($key.Replace('/','_')+'.obj')
        $command=[regex]::Replace([string]$entry.command,'(?<= -o )[^ ]+',('"'+$fresh+'"'))
        $command=[regex]::Replace($command,' -Xcompiler=-Fd[^ ]+','')
        $cudaCommands+=('pushd "'+$worker+'"')
        $cudaCommands+=$command
        $cudaCommands+='if errorlevel 1 exit /b 1'
        $cudaCommands+='popd'
        $rebuilt[$key]=[ordered]@{object=$fresh;command=$command}
        $objects+=('"'+$fresh+'"')
    }else{$objects+=('"'+$object+'"')}

}
$sources=@($all|Where-Object {$_ -notmatch '\.cu$'}|Select-Object -Unique)
$commands=@($cudaCommands)
foreach($source in $sources) {
    $object=Join-Path $output ($source.Replace('/','_')+'.obj');$objects+=('"'+$object+'"')
    $arch=if($source -match 'simd_|ifma'){ ' /arch:AVX512' }else{''}
    $commands+=('"'+$m.host_compiler+'" /nologo /TP /std:c++17 /O2 /EHsc /MD /utf-8 /wd4819 /DNOMINMAX /DECM_HAS_CUDA_STAGE1=1 /DECM_CUDA_ENABLE_PRAC=0'+$arch+
        ' /I "'+$repo+'/include" /I "'+$repo+'/src" /I "'+$m.gmp_root+'/include" /I "'+$m.cuda_root+'/include"'+
        ' /c "'+$source+'" /Fo"'+$object+'" /sourceDependencies "'+$object+'.deps.json"')
    $commands+='if errorlevel 1 exit /b 1'
    $closure=Get-EcmSourceClosure $repo @($source) @('include','src',(Join-Path $m.gmp_root 'include')) -Production
    foreach($dependency in $closure){$sourceHashes[$dependency]=(Get-FileHash -LiteralPath (Join-Path $repo $dependency)).Hash}
}
$exe=Join-Path $output 'ecm_cuda.exe'
$response=Join-Path $output 'link.rsp'
[IO.File]::WriteAllText($response,($objects -join "`r`n"),[Text.UTF8Encoding]::new($false))
$commands+=('"'+$m.host_compiler+'" /nologo @"'+$response+'" "'+$m.gmp_root+'/lib/gmp.lib" /Fe"'+$exe+'" /link /LIBPATH:"'+$m.cuda_root+'/lib/x64" cudadevrt.lib cudart.lib user32.lib advapi32.lib')
$commands+='exit /b %errorlevel%'
$batch=Join-Path $output 'build.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($m.compile_setup)`r`nif errorlevel 1 exit /b 1`r`n"+($commands -join "`r`n")+"`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Out-File -LiteralPath (Join-Path $output 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $output 'build.log') -Tail 35;throw 'Stage1 host overlay failed'}
foreach($object in $reused.Keys){if((Get-FileHash -LiteralPath $object).Hash -ne $reused[$object]){throw "Reused CUDA object changed: $object"}}
foreach($source in $sourceHashes.Keys){if((Get-FileHash -LiteralPath (Join-Path $repo $source)).Hash -ne $sourceHashes[$source]){throw "Source changed during build: $source"}}
Copy-Item -LiteralPath (Join-Path $m.gmp_root 'bin/gmp-10.dll') -Destination $output
[ordered]@{complete=$true;worker_manifest_sha256=(Get-FileHash -LiteralPath $manifestPath).Hash;
    reused_cuda=$reused;rebuilt_cuda=$rebuilt;source_hashes=$sourceHashes;cuda_compiler_invocations=$rebuilt.Count;
    exe_sha256=(Get-FileHash -LiteralPath $exe).Hash}|ConvertTo-Json -Depth 5|
    Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
Write-Host "Stage1 overlay built: host=$($sources.Count), verified CUDA=$($reused.Count), rebuilt CUDA=$($rebuilt.Count)"
