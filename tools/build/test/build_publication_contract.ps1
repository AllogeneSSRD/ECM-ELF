param([string]$WorkerBuild='build_stage1_local',
      [Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$manifest=Get-Content -LiteralPath (Join-Path $WorkerBuild 'stage1_local_manifest.json') -Raw|ConvertFrom-Json
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(Test-Path -LiteralPath $output){throw 'Use a fresh build directory'}
New-Item -ItemType Directory -Path $output|Out-Null
[IO.File]::WriteAllText((Join-Path $output '.gitignore'),"*`n")
$sources=@('tools/test/publication_contract.cpp','tools/test/adapters/publication_files.cpp',
 'src/io/results.cpp','src/io/save.cpp','src/ecm/application/configuration/ecm_expr.cpp',
 'src/io/stage1_checkpoint.cpp','src/ecm/platform/stage1_stop.cpp',
 'src/ecm/backend/opencl/stage1/montgomery/checkpoint.cpp','src/ecm/platform/logging.cpp',
 'src/ecm/application/configuration/runtime.cpp')
$commands=@();$objects=@();$hashes=[ordered]@{}
foreach($source in $sources){
    $object=Join-Path $output ($source.Replace('/','_')+'.obj');$objects+=('"'+$object+'"')
    $commands+=('"'+$manifest.host_compiler+'" /nologo /TP /std:c++17 /O2 /EHsc /MD /utf-8 /DNOMINMAX /I include /I src /I "'+$manifest.gmp_root+'/include" /c "'+$source+'" /Fo"'+$object+'"')
    $commands+='if errorlevel 1 exit /b 1'
    $hashes[$source]=(Get-FileHash -LiteralPath $source).Hash
}
$exe=Join-Path $output 'publication_contract.exe'
$commands+=('"'+$manifest.host_compiler+'" /nologo '+($objects -join ' ')+' "'+$manifest.gmp_root+'/lib/gmp.lib" /Fe"'+$exe+'"')
$commands+='exit /b %errorlevel%'
$batch=Join-Path $output 'build.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($manifest.compile_setup)`r`nif errorlevel 1 exit /b 1`r`n"+($commands -join "`r`n")+"`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Out-File -LiteralPath (Join-Path $output 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $output 'build.log') -Tail 35;throw 'Publication contract build failed'}
Copy-Item -LiteralPath (Join-Path $manifest.gmp_root 'bin/gmp-10.dll') -Destination $output
[ordered]@{source_hashes=$hashes;exe_sha256=(Get-FileHash -LiteralPath $exe).Hash;cuda_compiler_invocations=0}|
    ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
