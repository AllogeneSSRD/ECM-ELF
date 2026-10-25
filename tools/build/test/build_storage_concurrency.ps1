param([string]$WorkerBuild='build_stage1_local', [Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$m=Get-Content -LiteralPath (Join-Path $WorkerBuild 'stage1_local_manifest.json') -Raw|ConvertFrom-Json
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(Test-Path -LiteralPath $output){throw 'Use a fresh build directory'}
New-Item -ItemType Directory -Path $output|Out-Null
[IO.File]::WriteAllText((Join-Path $output '.gitignore'),"*`n")
$sources=@('tools/test/storage_concurrency_contract.cpp','src/io/worktodo.cpp',
    'src/io/files.cpp','src/ecm/core/stage1/ecm_stage1_exp.cpp')
$commands=@();$objects=@();$hashes=[ordered]@{}
foreach($source in $sources) {
    $object=Join-Path $output ($source.Replace('/','_')+'.obj')
    $objects+=('"'+$object+'"')
    $commands+=('"'+$m.host_compiler+'" /nologo /TP /std:c++17 /O2 /EHsc /MD /utf-8 /DNOMINMAX'+
        ' /I "'+$repo+'/include" /I "'+$repo+'/src" /I "'+$m.gmp_root+'/include"'+
        ' /c "'+$source+'" /Fo"'+$object+'"')
    $commands+='if errorlevel 1 exit /b 1'
    $hashes[$source]=(Get-FileHash -LiteralPath $source).Hash
}
foreach($source in @('src/io/exponent_cache.cpp','src/sync/stage2.cpp')) {
    $hashes[$source]=(Get-FileHash -LiteralPath $source).Hash
}
$exe=Join-Path $output 'storage_concurrency.exe'
$commands+=('"'+$m.host_compiler+'" /nologo '+($objects -join ' ')+' "'+$m.gmp_root+'/lib/gmp.lib" /Fe"'+$exe+'"')
$commands+='exit /b %errorlevel%'
$batch=Join-Path $output 'build.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($m.compile_setup)`r`nif errorlevel 1 exit /b 1`r`n"+($commands -join "`r`n")+"`r`n",[Text.UTF8Encoding]::new($false))
$timer=[Diagnostics.Stopwatch]::StartNew()
& cmd.exe /d /c $batch 2>&1|Out-File -LiteralPath (Join-Path $output 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $output 'build.log') -Tail 30;throw 'Storage contract build failed'}
Copy-Item -LiteralPath (Join-Path $m.gmp_root 'bin/gmp-10.dll') -Destination $output
[ordered]@{source_hashes=$hashes;exe_sha256=(Get-FileHash -LiteralPath $exe).Hash;build_seconds=$timer.Elapsed.TotalSeconds}|
    ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
