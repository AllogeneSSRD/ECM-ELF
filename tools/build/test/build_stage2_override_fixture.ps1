param([Parameter(Mandatory=$true)][string]$WorkerBuild,
      [Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$m=Get-Content -LiteralPath (Join-Path $WorkerBuild 'build_manifest.json') -Raw|ConvertFrom-Json
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(Test-Path -LiteralPath $output){throw 'Use a fresh build directory'}
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $output
$sources=@('tools/test/stage2_override_fixture.cpp','src/ecm/storage/results/publication.cpp',
    'src/ecm/storage/queue/stage2_progress.cpp','src/ecm/platform/fingerprint.cpp')
$command='"'+$m.host_compiler+'" /nologo /TP /std:c++17 /O2 /EHsc /MT /utf-8 /DNOMINMAX /I include /I src /I "'+
    $m.gmp_root+'/include" '+($sources -join ' ')+' /Fo"'+$output+'/" /Fe"'+$output+
    '/stage2_override_fixture.exe" /link /LIBPATH:"'+$m.gmp_root+'/lib" gmp.lib'
$batch=Join-Path $output 'build.cmd'
[IO.File]::WriteAllText($batch,[string]::Join([Environment]::NewLine,@('@echo off',$m.compile_setup,'if errorlevel 1 exit /b 1',$command,'exit /b %errorlevel%','')),[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Out-File -LiteralPath (Join-Path $output 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $output 'build.log') -Tail 25;throw 'Override fixture compile failed'}
Copy-Item -LiteralPath (Join-Path $m.gmp_root 'bin/gmp-10.dll') -Destination $output
[ordered]@{complete=$true;exe_sha256=(Get-FileHash -LiteralPath (Join-Path $output 'stage2_override_fixture.exe')).Hash;
    source_sha256=@($sources|ForEach-Object{[ordered]@{path=$_;sha256=(Get-FileHash -LiteralPath $_).Hash}})}|
    ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
