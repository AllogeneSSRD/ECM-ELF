param([string]$WorkerBuild='build_stage1_local',
      [Parameter(Mandatory=$true)][string]$Build,
      [Parameter(Mandatory=$true)][string]$Evidence)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$manifest=Get-Content -LiteralPath (Join-Path $WorkerBuild 'stage1_local_manifest.json') -Raw|ConvertFrom-Json
$buildPath=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
$evidencePath=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Evidence)
if(Test-Path -LiteralPath $buildPath){throw 'Use a fresh build directory'}
New-Item -ItemType Directory -Path $buildPath|Out-Null
[IO.File]::WriteAllText((Join-Path $buildPath ".gitignore"),"*`n")
$sources=@('tools/test/result_batch_contract.cpp','src/io/results.cpp',
 'src/io/files.cpp','src/io/save.cpp','src/ecm/application/configuration/ecm_expr.cpp','src/io/stage1_checkpoint.cpp','src/ecm/platform/stop.cpp','src/ecm/stage1/montgomery/opencl/checkpoint.cpp',
 'src/ecm/platform/logging.cpp','src/ecm/application/configuration/runtime.cpp')
$objects=@();$commands=@()
foreach($source in $sources){
 $object=Join-Path $buildPath ($source.Replace('/','_')+'.obj');$objects+=('"'+$object+'"')
 $commands+=('"'+$manifest.host_compiler+'" /nologo /TP /std:c++17 /O2 /EHsc /MD /utf-8 /I include /I src /I "'+$manifest.gmp_root+'/include" /c "'+$source+'" /Fo"'+$object+'"')
 $commands+='if errorlevel 1 exit /b 1'
}
$command=($commands -join "`r`n")+"`r`n"+'"'+$manifest.host_compiler+'" /nologo '+($objects -join ' ')+' /Fe"'+$buildPath+'/result_batch_contract.exe" /link /LIBPATH:"'+$manifest.gmp_root+'/lib" gmp.lib'
$batch=Join-Path $buildPath 'build.cmd'
$lines=@('@echo off',$manifest.compile_setup,'if errorlevel 1 exit /b 1',$command,'exit /b %errorlevel%','')
[IO.File]::WriteAllText($batch,[string]::Join([Environment]::NewLine,$lines),[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Tee-Object -FilePath (Join-Path $buildPath 'build.log')
if($LASTEXITCODE){throw 'Result batch contract build failed'}
Copy-Item -LiteralPath (Join-Path $manifest.gmp_root 'bin/gmp-10.dll') -Destination $buildPath
& (Join-Path $buildPath 'result_batch_contract.exe') $evidencePath 2>&1|Tee-Object -FilePath (Join-Path $buildPath 'test.log')
if($LASTEXITCODE){throw 'Result batch contract failed'}
[ordered]@{passed=$true;exe_sha256=(Get-FileHash -LiteralPath (Join-Path $buildPath 'result_batch_contract.exe')).Hash;
 source_sha256=@($sources|ForEach-Object{[ordered]@{path=$_;sha256=(Get-FileHash -LiteralPath $_).Hash}})}|
 ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $buildPath 'manifest.json') -Encoding utf8
