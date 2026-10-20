#Requires -Version 5.1
param([ValidateSet('stage1','stage2','tiers')][string]$Stage='stage1',
      [string]$Build='build_cuda_cmake/cuda_structure_p1',
      [string]$Tiers='384',[int]$Jobs=8,[switch]$EnablePrac)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
if($Stage -eq 'stage1') {
    & (Join-Path $PSScriptRoot '../build_stage1_local.ps1') -BuildDir "$Build/stage1" -Tiers $Tiers -Jobs $Jobs -EnablePrac:$EnablePrac -Incremental
} elseif($Stage -eq 'stage2') {
    & (Join-Path $PSScriptRoot '../build_stage2_local.ps1') -Build "$Build/stage2"
} else {
    . (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
    $cuda=Resolve-EcmCudaToolkit -Archs 89
    $vc=Resolve-EcmVcEnvironment $cuda
    $Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
    New-Item -ItemType Directory -Force $Build|Out-Null
    $all='128,192,256,384,512,768,1024,1280,1536,1792,2048,2560,3072,3584,4096,4608,5120,5632,6144,6656,7168,7680,8192,9216,10240,11264,12288,13312,14336,15360,16384,17408'
    $profiles=@(
        @('all','',$all),
        @('mont','/DECM_CUDA_MONT_ONLY',($all -replace ',17408$','')),
        @('dev','/DECM_CUDA_MONT_ONLY /DECM_CUDA_MONT_DEV_ONLY','128,192,256,384,512,768,1024'),
        @('restricted','/DECM_TIERS_RESTRICTED /DECM_TIER_384 /DECM_TIER_small /DECM_TIER_medium /DECM_TIER_17408','384,512,1024,17408'),
        @('numeric_aliases','/DECM_TIERS_RESTRICTED /DECM_TIER_512 /DECM_TIER_1024','512,1024'),
        @('empty','/DECM_TIERS_RESTRICTED',''),
        @('group0','/DECM_CUDA_TIER_GROUP=0','128,192,256,384,512'),
        @('group1','/DECM_CUDA_TIER_GROUP=1','768,1024,1280,1536,1792,2048'),
        @('group2','/DECM_CUDA_TIER_GROUP=2','2560,3072,3584,4096'),
        @('group3','/DECM_CUDA_TIER_GROUP=3','4608,5120,5632,6144'),
        @('group4','/DECM_CUDA_TIER_GROUP=4','6656,7168,7680,8192'),
        @('group5','/DECM_CUDA_TIER_GROUP=5','9216,10240,11264,12288'),
        @('group6','/DECM_CUDA_TIER_GROUP=6','13312,14336,15360,16384,17408'),
        @('prac_interval','/DECM_CUDA_MONT_ONLY /DECM_CUDA_TIER_MIN_BITS=9216 /DECM_CUDA_TIER_MAX_BITS=10240','9216,10240'),
        @('empty_intersection','/DECM_TIERS_RESTRICTED /DECM_TIER_384 /DECM_CUDA_TIER_MIN_BITS=9216',''),
        @('mont_tpb_override','/DECM_TPB=256',$all))
    foreach($profile in $profiles){
        $exe=Join-Path $Build ($profile[0]+'.exe');$obj=Join-Path $Build ($profile[0]+'.obj')
        $command='"'+$vc.Compiler+'" /nologo /O2 /EHsc /std:c++17 /I include '+$profile[1]+ ' tools/test/cuda_tiers_test.cpp /Fo"'+$obj+'" /Fe"'+$exe+'"'
        $batch=Join-Path $Build ($profile[0]+'.cmd')
        [IO.File]::WriteAllText($batch,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
        & cmd.exe /d /c $batch
        if($LASTEXITCODE -ne 0){throw "Tier profile compile failed: $($profile[0])"}
        & $exe ([string]$profile[2])
        if($LASTEXITCODE -ne 0){throw "Tier profile failed: $($profile[0])"}
    }
    [ordered]@{complete=$true;profiles=$profiles.Count;gpu_runs=0}|ConvertTo-Json|Set-Content (Join-Path $Build 'result.json') -Encoding utf8
    Write-Host "PASS $($profiles.Count) tier build profiles"
}
if($LASTEXITCODE -ne 0){throw "P1 validation failed: $Stage"}
