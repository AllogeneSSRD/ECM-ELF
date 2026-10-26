param([string]$WorkerBuild='build_stage1_local',[Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$m=Get-Content -LiteralPath (Join-Path $WorkerBuild 'stage1_local_manifest.json') -Raw|ConvertFrom-Json
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(Test-Path -LiteralPath $output){throw 'Use a fresh build directory'}
New-Item -ItemType Directory -Path $output|Out-Null
[IO.File]::WriteAllText((Join-Path $output '.gitignore'),"*`n")
$sources=@('tools/test/shared_writer_contract.cpp','tools/test/adapters/shared_writer_prime95.cpp',
 'tools/test/adapters/shared_writer_results.cpp','tools/test/adapters/shared_writer_gui.cpp',
 'src/io/files.cpp','src/io/worktodo.cpp','src/io/save.cpp','src/sync/p95.cpp',
 'src/ecm/stage2/input.cpp','src/ecm/application/configuration/ecm_expr.cpp',
 'src/ecm/platform/logging.cpp','src/ecm/application/configuration/runtime.cpp')
$commands=@();$objects=@();$hashes=[ordered]@{}
foreach($source in $sources){
 $object=Join-Path $output ($source.Replace('/','_')+'.obj');$objects+=('"'+$object+'"')
 $commands+=('"'+$m.host_compiler+'" /nologo /TP /std:c++17 /O2 /EHsc /MD /utf-8 /DNOMINMAX /I include /I src /I "'+$m.gmp_root+'/include" /c "'+$source+'" /Fo"'+$object+'" /sourceDependencies "'+$object+'.deps.json"')
 $commands+='if errorlevel 1 exit /b 1'
 $hashes[$source]=(Get-FileHash -LiteralPath $source).Hash
}
$exe=Join-Path $output 'shared_writer_contract.exe'
$commands+=('"'+$m.host_compiler+'" /nologo '+($objects -join ' ')+' "'+$m.gmp_root+'/lib/gmp.lib" /Fe"'+$exe+'"')
$commands+='exit /b %errorlevel%'
$batch=Join-Path $output 'build.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($m.compile_setup)`r`nif errorlevel 1 exit /b 1`r`n"+($commands -join "`r`n")+"`r`n",[Text.UTF8Encoding]::new($false))
$timer=[Diagnostics.Stopwatch]::StartNew()
& cmd.exe /d /c $batch 2>&1|Out-File -LiteralPath (Join-Path $output 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $output 'build.log') -Tail 30;throw 'Writer contract build failed'}
Copy-Item -LiteralPath (Join-Path $m.gmp_root 'bin/gmp-10.dll') -Destination $output
$headers=[ordered]@{}
foreach($object in $objects){
 $deps=Get-Content -LiteralPath ($object.Trim('"')+'.deps.json') -Raw|ConvertFrom-Json
 foreach($path in $deps.Data.Includes){$headers[$path]=(Get-FileHash -LiteralPath $path).Hash}
}
[ordered]@{source_hashes=$hashes;header_hashes=$headers;exe_sha256=(Get-FileHash -LiteralPath $exe).Hash;build_seconds=$timer.Elapsed.TotalSeconds;gpu_curves=0}|
 ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
