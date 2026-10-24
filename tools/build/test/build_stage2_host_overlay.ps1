# Link an isolated Stage2 test executable with verified unchanged production
# objects. Recompile changed host units; never alter the worker build/cache.
param([Parameter(Mandatory=$true)][string]$WorkerBuild,
      [Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$worker=(Resolve-Path -LiteralPath $WorkerBuild).Path
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(Test-Path -LiteralPath $output){throw 'Use a fresh test build directory'}
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
foreach($property in $m.compile_units.PSObject.Properties) {
    $source=$property.Name;$unit=$property.Value
    $state=Get-EcmCompileState $repo $unit.entry $m.compile_setup -Production
    $states[$source]=$state
    $unchanged=$state.fingerprint -eq $unit.fingerprint -and
        (Test-Path -LiteralPath $state.object) -and (Get-FileHash -LiteralPath $state.object).Hash -eq $unit.object_sha256
    if($unchanged){$reused[$source]=[ordered]@{object=$state.object;sha256=$unit.object_sha256};continue}
    if($source.EndsWith('.cu')){throw "CUDA inputs changed; build the production worker normally first: $source"}
    $object=Join-Path $output ($source.Replace('/','_')+'.obj')
    $command=$unit.entry.command.Replace($state.object,$object)
    if($command -eq $unit.entry.command){throw "Could not redirect host object: $source"}
    $commands+=$command;$commands+='if errorlevel 1 exit /b 1'
    $entry=$unit.entry|ConvertTo-Json -Depth 5|ConvertFrom-Json
    $entry.command=$command;$entry.output=$object;$entries[$source]=$entry
    $link=$link.Replace($state.object,$object)
}
New-Item -ItemType Directory -Path $output|Out-Null
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
    $entry=if($entries.Contains($source)){$entries[$source]}else{$m.compile_units.$source.entry}
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
