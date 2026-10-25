# Build the complete Stage1 OpenCL program with CUDA targets disabled.
param([string]$WorkerBuild='build_stage1_local',
      [Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
$worker=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkerBuild)
$output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
New-Item -ItemType Directory -Force -Path $output | Out-Null
[IO.File]::WriteAllText((Join-Path $output '.gitignore'),"*`n")

. (Join-Path $PSScriptRoot '../internal/cuda_build_state.ps1')
# Freeze project source identity for this complete CMake build.
$sourcePaths=@(Get-ChildItem -LiteralPath (Join-Path $repo 'src'),(Join-Path $repo 'include') -Recurse -File |
    Where-Object {$_.Extension -in @('.cpp','.c','.h','.hpp','.cu','.cuh','.inl')} |
    ForEach-Object {Get-EcmSourceKey $repo $_.FullName})+@('CMakeLists.txt','tools/build/internal/cuda_sources.cmake')
$sourceHashes=Get-EcmSourceHashes $repo $sourcePaths
$manifest=Get-Content -LiteralPath (Join-Path $worker 'stage1_local_manifest.json') -Raw | ConvertFrom-Json
$sslEntry=Get-Content -LiteralPath (Join-Path $worker 'CMakeCache.txt') |
    Where-Object { $_ -match '^OPENSSL_INCLUDE_DIR:PATH=' } | Select-Object -First 1
if(-not $sslEntry){throw 'Worker CMake cache must identify the OpenSSL installation'}
$sslRoot=Split-Path -Parent ($sslEntry -replace '^OPENSSL_INCLUDE_DIR:PATH=','')
$compiler=$manifest.host_compiler.Replace('\','/')
$gmp=$manifest.gmp_root.Replace('\','/')
$cuda=$manifest.cuda_root.Replace('\','/')
$targets=@('ecm')
$configure='cmake -S "'+$repo+'" -B "'+$output+'" -G "NMake Makefiles"'+
    ' -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="'+$compiler+'" -DCMAKE_CXX_COMPILER="'+$compiler+'"'+
    ' -DECM_ENABLE_CUDA=OFF -DCMAKE_CUDA_COMPILER=NOTFOUND -DECM_BUILD_TOOLS=OFF'+
    ' -DECM_BUILD_GUI=OFF -DECM_BUILD_GUI_V2=OFF -DECM_BUILD_CPU_STAGE2=OFF'+
    ' -DECM_WINDOWS_GMP_ROOT="'+$gmp+'" -DOPENSSL_ROOT_DIR="'+$sslRoot+'"'+
    ' -DOpenCL_INCLUDE_DIR="'+$cuda+'/include" -DOpenCL_LIBRARY="'+$cuda+'/lib/x64/OpenCL.lib"'
$batch=Join-Path $output 'build.cmd'
$commands=@('@echo off',$manifest.compile_setup,'if errorlevel 1 exit /b 1',
    $configure,'if errorlevel 1 exit /b 1',
    ('cmake --build "'+$output+'" --target '+($targets -join ' ')),
    'exit /b %errorlevel%')
[IO.File]::WriteAllText($batch,($commands -join "`r`n")+"`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $batch 2>&1 | Out-File -LiteralPath (Join-Path $output 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content -LiteralPath (Join-Path $output 'build.log') -Tail 35;throw 'OpenCL Stage1 build failed'}

Assert-EcmHashes $sourceHashes (Get-EcmSourceHashes $repo $sourcePaths) 'Source changed during OpenCL build'
$hashes=[ordered]@{}
foreach($target in $targets){
    $exe=Join-Path $output ($target+'.exe')
    $hashes[$target]=(Get-FileHash -LiteralPath $exe).Hash
}
[ordered]@{complete=$true;targets=$hashes;source_hashes=$sourceHashes;cuda_compiler_invocations=0;gpu_runs=0} |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $output 'stage1_opencl_manifest.json') -Encoding utf8
Write-Host "Built complete OpenCL Stage1; no CUDA compilation or GPU execution."
