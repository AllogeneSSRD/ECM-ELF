#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/edwards_gpu_stage1_probe',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89',
      [switch]$Incremental)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
$cuda=Resolve-EcmCudaToolkit -Archs $Arch
$vc=Resolve-EcmVcEnvironment $cuda
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
New-Item -ItemType Directory -Force $Build | Out-Null
$sources=@('src/cuda/ecm_cuda_edwards.cu','tools/test/edwards_gpu_stage1_probe.cpp','src/cpu/ecm_edwards_cpu.cpp')
$objects=@()
$previous=$null
$manifestPath=Join-Path $Build 'manifest.json'
if($Incremental -and (Test-Path $manifestPath)){$previous=Get-Content $manifestPath -Raw|ConvertFrom-Json}
foreach($source in $sources){
    $stem=[IO.Path]::GetFileNameWithoutExtension($source)
    $obj=Join-Path $Build ($stem+'.obj');$objects+=$obj
    $command='"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" -std=c++17 -O2 -arch='+$Arch+
        ' --pre-include=stdint.h -Xcompiler /utf-8 -Xcompiler /wd4819 -I cgbn/include -I third_party/gmp-zen3/dist/include -I src/core -I src/cpu -I src/cuda -c "'+$source+'" -o "'+$obj+'"'
    $cmdFile=Join-Path $Build ($stem+'.cmd')
    $commandText="@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n"
    $reuse=$false
    if($source -like '*.cu' -and $previous -and $previous.arch -eq $Arch -and (Test-Path $obj) -and (Test-Path $cmdFile)){
        $reuse=[IO.File]::ReadAllText($cmdFile) -eq $commandText
        foreach($dep in @($source,'src/cuda/ecm_cuda_edwards.h','kernels/cuda/cgbn_edwards_point.cuh')){
            if($previous.sources.$dep -ne (Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash){$reuse=$false}
        }
        if($previous.cuda_object_sha256 -and $previous.cuda_object_sha256 -ne (Get-FileHash $obj).Hash){$reuse=$false}
    }
    if($reuse){Write-Host "Reused unchanged GPU translation unit: $source";continue}
    [IO.File]::WriteAllText($cmdFile,$commandText,[Text.UTF8Encoding]::new($false))
    & cmd.exe /d /c $cmdFile 2>&1 | Out-File (Join-Path $Build ($stem+'.log')) -Encoding utf8
    if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build ($stem+'.log')) -Tail 30;throw "compile failed: $source"}
    Write-Host "Compiled $source"
}
$objArgs=($objects|ForEach-Object{'"'+$_+'"'}) -join ' '
$exe=Join-Path $Build 'edwards_gpu_stage1_probe.exe'
$command='"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" -arch='+$Arch+' '+$objArgs+' -L third_party/gmp-zen3/dist/lib -lgmp -o "'+$exe+'"'
$cmdFile=Join-Path $Build 'link.cmd'
[IO.File]::WriteAllText($cmdFile,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $cmdFile 2>&1 | Out-File (Join-Path $Build 'link.log') -Encoding utf8
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'link.log') -Tail 30;throw 'link failed'}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
$hashes=[ordered]@{}
foreach($source in ($sources+@('src/core/ecm_edwards_curve.h','src/core/ecm_edwards_io.h','src/cuda/ecm_cuda_edwards.h','kernels/cuda/cgbn_edwards_point.cuh','src/cpu/ecm_edwards_mont.h','tools/build/test/build_edwards_gpu_stage1_probe.ps1'))){
    $hashes[$source]=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
}
[ordered]@{exe=$exe;sha256=(Get-FileHash $exe).Hash;arch=$Arch;
    cuda_object_sha256=(Get-FileHash (Join-Path $Build 'ecm_cuda_edwards.obj')).Hash;
    sources=$hashes}|ConvertTo-Json -Depth 5|Set-Content $manifestPath -Encoding utf8
Write-Host "Built $exe"
