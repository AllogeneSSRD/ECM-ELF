# Host-only capacity/ownership gate, using the formal build's verified compiler.
param([string]$WorkerBuild='build_stage2_local',[string]$Build='build_dev/stage2_pinned_reserve_fixture')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$worker=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkerBuild)
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
if(Test-Path -LiteralPath $output){throw 'Use a fresh output directory'}
$m=Get-Content -LiteralPath (Join-Path $worker 'build_manifest.json') -Raw -Encoding UTF8|ConvertFrom-Json
if(-not $m.host_compiler -or -not $m.compile_setup){throw 'Formal host compiler identity required'}
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $output
$source='tools/test/stage2_pinned_reserve_fixture.cpp'
$exe=Join-Path $output 'stage2_pinned_reserve_fixture.exe'
$command='"'+$m.host_compiler+'" /nologo /std:c++17 /O2 /EHsc /MT /utf-8 /DNOMINMAX "'+$source+'" /Fo"'+$output+'/fixture.obj" /Fe"'+$exe+'"'
$batch=Join-Path $output 'build.cmd'
[IO.File]::WriteAllText($batch,[string]::Join([Environment]::NewLine,@('@echo off',$m.compile_setup,'if errorlevel 1 exit /b 1',$command,'exit /b %errorlevel%','')),[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1|Out-File -LiteralPath (Join-Path $output 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $output 'build.log') -Tail 20;throw 'Pinned reserve fixture compile failed'}
& $exe | Tee-Object -FilePath (Join-Path $output 'result.json')
if($LASTEXITCODE){throw 'Pinned reserve fixture failed'}
[ordered]@{complete=$true;exe_sha256=(Get-FileHash -LiteralPath $exe).Hash;cuda_calls=0;
    source_sha256=(Get-FileHash -LiteralPath $source).Hash;
    header_sha256=(Get-FileHash -LiteralPath 'src/ecm/stage2/cuda/pinned_reserve.h').Hash;
    snapshot_header_sha256=(Get-FileHash -LiteralPath 'src/ecm/stage2/cuda/snapshot_pinned.h').Hash}|ConvertTo-Json|
    Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
