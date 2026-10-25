param([string]$WorkerBuild='build_stage1_local',[Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$m=Get-Content -LiteralPath (Join-Path $WorkerBuild 'stage1_local_manifest.json') -Raw|ConvertFrom-Json
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
New-Item -ItemType Directory -Force -Path $output|Out-Null
[IO.File]::WriteAllText((Join-Path $output '.gitignore'),"*`n")
$sources=@('src/io/worktodo.cpp','src/io/files.cpp','src/io/results.cpp','src/sync/p95.cpp','src/sync/backup.cpp',
    'src/io/save.cpp','src/ecm/application/configuration/ecm_expr.cpp',
    'src/io/prime95.cpp','src/ecm/platform/logging.cpp',
    'src/ecm/application/configuration/runtime.cpp','tools/test/stage1_host_io_contract.cpp')
$commands=@();$objects=@();$hashes=[ordered]@{}
foreach($source in $sources) {
    $object=Join-Path $output ($source.Replace('/','_')+'.obj');$objects+=('"'+$object+'"')
    $commands+=('"'+$m.host_compiler+'" /nologo /TP /std:c++17 /O2 /EHsc /MD /utf-8 /DNOMINMAX'+
        ' /I "'+$repo+'/include" /I "'+$repo+'/src" /I "'+$m.gmp_root+'/include"'+
        ' /c "'+$source+'" /Fo"'+$object+'" /sourceDependencies "'+$object+'.deps.json"')
    $commands+='if errorlevel 1 exit /b 1';$hashes[$source]=(Get-FileHash -LiteralPath $source).Hash
}
$exe=Join-Path $output 'stage1_host_io_contract.exe'
$commands+=('"'+$m.host_compiler+'" /nologo '+($objects -join ' ')+' "'+$m.gmp_root+'/lib/gmp.lib" /Fe"'+$exe+'"')
$commands+='exit /b %errorlevel%'
$batch=Join-Path $output 'build.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($m.compile_setup)`r`nif errorlevel 1 exit /b 1`r`n"+($commands -join "`r`n")+"`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Out-File -LiteralPath (Join-Path $output 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $output 'build.log') -Tail 30;throw 'Host I/O contract build failed'}
Copy-Item -LiteralPath (Join-Path $m.gmp_root 'bin/gmp-10.dll') -Destination $output
[ordered]@{complete=$true;source_hashes=$hashes;cuda_compiler_invocations=0;exe_sha256=(Get-FileHash -LiteralPath $exe).Hash}|
    ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
