#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/edwards_gpu_workers/gui_metadata')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
$cuda=Resolve-EcmCudaToolkit -Archs 89
$vc=Resolve-EcmVcEnvironment $cuda
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
New-Item -ItemType Directory -Force $Build | Out-Null
$sources=@('tools/test/edwards_gui_metadata_probe.cpp','src/gui/log_parse.cpp','src/gui/results.cpp')
$objects=@()
foreach($source in $sources){
    $stem=[IO.Path]::GetFileNameWithoutExtension($source);$obj=Join-Path $Build ($stem+'.obj');$objects+=$obj
    $command='"'+$vc.Compiler+'" /nologo /O2 /std:c++17 /EHsc /utf-8 /I src/gui /c "'+$source+'" /Fo"'+$obj+'"'
    $file=Join-Path $Build ($stem+'.cmd')
    [IO.File]::WriteAllText($file,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
    & cmd.exe /d /c $file 2>&1 | Out-File (Join-Path $Build ($stem+'.log')) -Encoding utf8
    if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build ($stem+'.log')) -Tail 30;throw "compile failed: $source"}
}
$exe=Join-Path $Build 'edwards_gui_metadata_probe.exe'
$argsText=($objects|ForEach-Object{'"'+$_+'"'}) -join ' '
$command='"'+$vc.Compiler+'" /nologo '+$argsText+' /Fe"'+$exe+'"'
$file=Join-Path $Build 'link.cmd'
[IO.File]::WriteAllText($file,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $file 2>&1 | Out-File (Join-Path $Build 'link.log') -Encoding utf8
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'link.log') -Tail 30;throw 'link failed'}
$hashes=[ordered]@{}
foreach($source in ($sources+@('src/gui/log_parse.h','src/gui/results.h','tools/build/test/build_edwards_gui_metadata_probe.ps1'))){
    $hashes[$source]=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
}
[ordered]@{exe=$exe;sha256=(Get-FileHash $exe).Hash;sources=$hashes}|ConvertTo-Json -Depth 5|Set-Content (Join-Path $Build 'manifest.json') -Encoding utf8
Write-Host "Built $exe"
