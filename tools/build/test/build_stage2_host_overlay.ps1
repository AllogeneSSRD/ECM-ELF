# Link an isolated Stage2 test executable with verified unchanged production
# objects. Recompile changed host units; never alter the worker build/cache.
param([Parameter(Mandatory=$true)][string]$WorkerBuild,
      [Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$worker=(Resolve-Path -LiteralPath $WorkerBuild).Path
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
New-Item -ItemType Directory -Force -Path $output|Out-Null
[IO.File]::WriteAllText((Join-Path $output '.gitignore'),"*`n")
$manifestPath=Join-Path $worker 'build_manifest.json'
$m=Get-Content -LiteralPath $manifestPath -Raw|ConvertFrom-Json
if($m.engine -ne 'production' -or -not $m.compile_units -or -not $m.link_inputs.command){throw 'Verified production build manifest required'}
. (Join-Path $PSScriptRoot '../internal/cuda_build_state.ps1')
$states=[ordered]@{};$reused=[ordered]@{};$commands=@();$entries=[ordered]@{}
$link=$m.link_inputs.command
$prefix=$m.compile_setup+' && '
if(-not $link.StartsWith($prefix)){throw 'Unexpected recorded link setup'}
$link=$link.Substring($prefix.Length)
# Validate every CUDA unit before preparing or compiling host output.
$moves=@{'src/ecm/storage/queue/ecm_worktodo.cpp'='src/io/worktodo.cpp';
    'src/ecm/application/configuration/ecm_queue_config.cpp'='src/io/config.cpp'}
$moves['src/ecm/storage/cache/ecm_stage1_exp_cache.cpp']='src/io/exponent_cache.cpp'
$moves['src/ecm/storage/checkpoint/ecm_checkpoint.cpp']='src/ecm/stage1/montgomery/opencl/checkpoint.cpp'
$moves['src/ecm/storage/checkpoint/ecm_mont_ckpt.cpp']='src/ecm/stage1/montgomery/cpu/checkpoint.cpp'
$moves['src/ecm/storage/save/ecm_edwards_save.cpp']='src/io/prime95_save.cpp'
$moves['src/ecm/storage/queue/p95_worktodo.cpp']='src/io/prime95.cpp'
$moves['src/ecm/storage/queue/stage2_progress.cpp']='src/io/stage2_checkpoint.cpp'
$moves['src/ecm/storage/transfer/ecm_stage2_transfer.cpp']='src/sync/stage2.cpp'
$moves['src/ecm/storage/transfer/p95_transfer.cpp']='src/sync/p95.cpp'
$moves['src/sync/files.cpp']='src/io/files.cpp'
$moves['src/ecm/storage/checkpoint/ecm_stage1_checkpoint_runtime.cpp']='src/io/stage1_checkpoint.cpp'
$moves['src/ecm/storage/save/ecm_save.cpp']='src/io/save.cpp'
$moves['src/ecm/storage/save/publication.cpp']='src/io/save.cpp'
$moves['src/ecm/storage/results/publication.cpp']='src/io/results.cpp'
$units=[ordered]@{}
foreach($property in $m.compile_units.PSObject.Properties) {
    $old=$property.Name;$source=if($moves.ContainsKey($old)){$moves[$old]}else{$old}
    $unit=$property.Value|ConvertTo-Json -Depth 8|ConvertFrom-Json
    if($source -ne $old){$unit.entry.file=Join-Path $repo $source;$unit.entry.command=$unit.entry.command.Replace($old,$source).Replace($old.Replace('/','\'),$source.Replace('/','\'))}
    $units[$source]=$unit
}
$inventory=Get-EcmCudaInventory $repo
foreach($source in $inventory.groups.STAGE2_HOST) {
    if($units.Contains($source)){continue}
    $unit=$units['src/io/config.cpp']|ConvertTo-Json -Depth 8|ConvertFrom-Json
    $unit.entry.file=Join-Path $repo $source
    $unit.entry.command=$unit.entry.command.Replace('src/io/config.cpp',$source)
    $unit.fingerprint='';$units[$source]=$unit
    $link+=' "'+(Join-Path $output ($source.Replace('/','_')+'.obj'))+'"'
}
foreach($source in $units.Keys) {
    $unit=$units[$source]
    $linkObject=Get-EcmObjectPath $unit.entry
    if(-not $source.EndsWith('.cu')){
        $newObject=Join-Path $output ($source.Replace('/','_')+'.obj')
        $unit.entry.command=$unit.entry.command.Replace($linkObject,$newObject).Replace(' /c ',' /I "'+$repo+'/src" /c ')
        $unit.entry.output=$newObject
    }
    $state=Get-EcmCompileState $repo $unit.entry $m.compile_setup -Production
    $states[$source]=$state
    $unchanged=$state.fingerprint -eq $unit.fingerprint -and
        (Test-Path -LiteralPath $state.object) -and (Get-FileHash -LiteralPath $state.object).Hash -eq $unit.object_sha256
    if($unchanged){$reused[$source]=[ordered]@{object=$state.object;sha256=$unit.object_sha256};continue}
    if($source.EndsWith('.cu')){throw "CUDA inputs changed; build the production worker normally first: $source"}
    $object=Join-Path $output ($source.Replace('/','_')+'.obj')
    $command=$unit.entry.command.Replace($state.object,$object)
    if($state.object -ne $object){throw "Could not redirect host object: $source"}
    $commands+=$command;$commands+='if errorlevel 1 exit /b 1'
    $entry=$unit.entry|ConvertTo-Json -Depth 5|ConvertFrom-Json
    $entry.command=$command;$entry.output=$object;$entries[$source]=$entry
    $link=$link.Replace($linkObject,$object)
}
$exe=Join-Path $output 'ecm_cuda_stage2.exe'
$link=$link.Replace((Join-Path $worker 'ecm_cuda_stage2.exe'),$exe)
if(-not $link.Contains($exe)){throw 'Could not redirect test executable'}
$commands+=$link;$commands+='exit /b %errorlevel%';$commands+=''
$batch=Join-Path $output 'build.cmd'
$lines=@('@echo off',$m.compile_setup,'if errorlevel 1 exit /b 1')+$commands
[IO.File]::WriteAllText($batch,[string]::Join([Environment]::NewLine,$lines),[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Out-File -LiteralPath (Join-Path $output 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $output 'build.log') -Tail 30;throw 'Stage2 host overlay failed'}
foreach($source in $states.Keys) {
    $entry=if($entries.Contains($source)){$entries[$source]}else{$units[$source].entry}
    $after=Get-EcmCompileState $repo $entry $m.compile_setup -Production
    Assert-EcmCompileInputs $states[$source] $after
    if($reused.Contains($source) -and (Get-FileHash -LiteralPath $reused[$source].object).Hash -ne $reused[$source].sha256){throw "Reused object changed: $source"}
}
Copy-Item -LiteralPath (Join-Path $m.gmp_root 'bin/gmp-10.dll') -Destination $output
$sources=[ordered]@{}
foreach($state in $states.Values){foreach($key in $state.source_hashes.Keys){$sources[$key]=$state.source_hashes[$key]}}
[ordered]@{complete=$true;worker=$worker;worker_manifest_sha256=(Get-FileHash -LiteralPath $manifestPath).Hash;
    reused=$reused;rebuilt=@($entries.Keys);source_hashes=$sources;cuda_compiler_invocations=0;
    exe_sha256=(Get-FileHash -LiteralPath $exe).Hash}|
    ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
Write-Host "Stage2 overlay built: host rebuilt=$($entries.Count), verified reused=$($reused.Count), CUDA compiles=0"
