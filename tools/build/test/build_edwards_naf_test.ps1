#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/edwards_naf_test')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
$cuda=Resolve-EcmCudaToolkit -Archs 89
$vc=Resolve-EcmVcEnvironment $cuda
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
New-Item -ItemType Directory -Force $Build | Out-Null
$exe=Join-Path $Build 'edwards_naf_test.exe';$obj=Join-Path $Build 'edwards_naf_test.obj'
$command='"'+$vc.Compiler+'" /nologo /O2 /std:c++17 /EHsc /utf-8 /I include /I src /I third_party/gmp-zen3/dist/include tools/test/edwards_naf_test.cpp /Fo"'+$obj+'" /Fe"'+$exe+'" /link /LIBPATH:third_party/gmp-zen3/dist/lib gmp.lib'
$file=Join-Path $Build 'build.cmd'
[IO.File]::WriteAllText($file,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $file 2>&1 | Out-File (Join-Path $Build 'build.log') -Encoding utf8
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 30;throw 'compile failed'}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
$hashes=[ordered]@{}
foreach($source in @('tools/test/edwards_naf_test.cpp','src/ecm/core/stage1/ecm_edwards_naf.h','src/ecm/backend/cuda/stage1/edwards/edwards_slice.h','tools/build/test/build_edwards_naf_test.ps1')){
    $hashes[$source]=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
}
[ordered]@{exe=$exe;sha256=(Get-FileHash $exe).Hash;sources=$hashes}|ConvertTo-Json -Depth 5|Set-Content (Join-Path $Build 'manifest.json') -Encoding utf8
& $exe
if($LASTEXITCODE -ne 0){throw 'NAF CPU gate failed'}
