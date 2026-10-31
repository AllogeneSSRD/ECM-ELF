# Reuse an audited host overlay. Only the diagnostic options adapter is rebuilt;
# all device objects and the remaining host objects are checked before/after.
param([Parameter(Mandatory=$true)][string]$Overlay,
      [Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$base=(Resolve-Path -LiteralPath $Overlay).Path
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(Test-Path -LiteralPath $output){throw 'Use a fresh probe output directory'}
New-Item -ItemType Directory -Path $output|Out-Null
[IO.File]::WriteAllText((Join-Path $output '.gitignore'),"*`n")
$manifest=Get-Content -LiteralPath (Join-Path $base 'manifest.json') -Raw|ConvertFrom-Json
if(-not $manifest.complete){throw 'Need a complete host overlay'}
foreach($inputFile in $manifest.source_hashes.PSObject.Properties) {
    $inputPath=if([IO.Path]::IsPathRooted($inputFile.Name)){$inputFile.Name}else{Join-Path $repo $inputFile.Name}
    if((Get-FileHash -LiteralPath $inputPath).Hash -ine $inputFile.Value){throw "Overlay source changed: $($inputFile.Name)"}
}
$lines=Get-Content -LiteralPath (Join-Path $base 'build.cmd')
$compile=@($lines|Where-Object {$_ -like '*/c "src/ecm/stage2/cuda/options.cpp"*'})
$link=@($lines|Where-Object {$_ -like '*-o "*ecm_cuda_stage2.exe"'})
if($compile.Count -ne 1 -or $link.Count -ne 1){throw 'Unknown overlay command structure'}
$oldObject=Join-Path $base 'src_ecm_stage2_cuda_options.cpp.obj'
$newObject=Join-Path $output 'point_routes_options.obj'
$adapter=Join-Path $repo 'tools/experiments/cuda/stage2_point_routes_host.cpp'
$command=$compile[0].Replace('"src/ecm/stage2/cuda/options.cpp"','"'+$adapter+'"').Replace($oldObject,$newObject)
$exe=Join-Path $output 'ecm_cuda_stage2.exe'
$linkCommand=$link[0].Replace($oldObject,$newObject).Replace((Join-Path $base 'ecm_cuda_stage2.exe'),$exe)
if(-not $linkCommand.Contains($exe) -or -not $command.Contains($newObject)){throw 'Could not isolate output'}
$objects=[ordered]@{}
foreach($match in [regex]::Matches($link[0],'"([^"]+\.obj)"')) {
    $path=$match.Groups[1].Value
    if($path -ne $oldObject){$objects[$path]=(Get-FileHash -LiteralPath $path).Hash}
}
$batch=Join-Path $output 'build.cmd'
$commands=@('@echo off',$lines[1],'if errorlevel 1 exit /b 1',$command,'if errorlevel 1 exit /b 1',$linkCommand,'exit /b %errorlevel%')
[IO.File]::WriteAllText($batch,[string]::Join([Environment]::NewLine,$commands),[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Out-File -LiteralPath (Join-Path $output 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $output 'build.log') -Tail 20;throw 'Point policy probe failed'}
foreach($path in $objects.Keys){if((Get-FileHash -LiteralPath $path).Hash -ne $objects[$path]){throw "Reused object changed: $path"}}
Copy-Item -LiteralPath (Join-Path $base 'gmp-10.dll') -Destination $output
[ordered]@{complete=$true;kind='isolated_point_route_probe';base=$base;
    base_exe_sha256=(Get-FileHash -LiteralPath (Join-Path $base 'ecm_cuda_stage2.exe')).Hash;
    exe_sha256=(Get-FileHash -LiteralPath $exe).Hash;reused_objects=$objects;
    adapter_sha256=(Get-FileHash -LiteralPath $adapter).Hash;cuda_compiles=0;
    chain_blocks=@(32,64,128);production_unchanged=$true}|
    ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
Write-Host "Point route probe: one host adapter; reused $($objects.Count) objects; CUDA compiles=0"
