#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/stage1_host_wait_probe',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89',
      [ValidateSet(1280,5632,10240)][int]$Tier=1280)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot));Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
. (Join-Path $PSScriptRoot '../internal/cuda_build_state.ps1')
$cuda=Resolve-EcmCudaToolkit -Archs $Arch;$vc=Resolve-EcmVcEnvironment $cuda
$Build=Resolve-EcmSourcePath $repo $Build
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
$kernels=@{1280='cgbn_edwards_instances_tpi8.cu';5632='cgbn_edwards_instances_tpi16_4608_6144.cu';10240='cgbn_edwards_instances_tpi32_9216_12288.cu'}
$sources=@('tools/test/stage1_host_wait_probe.cu','tools/test/adapters/edwards_host_wait.cu',
    ('src/ecm/stage1/edwards/cuda/instances/'+$kernels[$Tier]),'src/ecm/stage1/scalar.cpp')
$dirs=@('include','third_party/cgbn/include','third_party/gmp-zen3/dist/include')
$deps=@(Get-EcmSourceClosure $repo $sources $dirs)+@('tools/build/test/build_stage1_host_wait_probe.ps1',
    'tools/build/internal/cuda_build_state.ps1','tools/build/internal/cuda_toolchain.ps1')
$hashes=Get-EcmSourceHashes $repo $deps;$names=Get-EcmObjectNames $sources
$commands=@();$objects=@()
foreach($source in $sources) {
    $object=Join-Path $Build ($names[$source]+'.obj');$objects+=$object
    $commands+=('"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" -std=c++17 -O2 -arch='+$Arch+
        ' -DECM_HOST_WAIT_TIER='+$Tier+' -DECM_TIERS_RESTRICTED -DECM_TIER_'+$Tier+
        ' --pre-include=stdint.h -Xcompiler /MD -Xcompiler /utf-8 -Xcompiler /wd4819 '+
        (($dirs|ForEach-Object {'-I "'+(Resolve-EcmSourcePath $repo $_)+'"'}) -join ' ')+
        ' -c "'+$source+'" -o "'+$object+'"')
    $commands+='if errorlevel 1 exit /b 1'
}
$exe=Join-Path $Build 'probe.exe'
$commands+=('"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" -Xcompiler /MD '+
    (($objects|ForEach-Object {'"'+$_+'"'}) -join ' ')+
    ' -L third_party/gmp-zen3/dist/lib -lgmp -o "'+$exe+'"')
$batch=Join-Path $Build 'build.cmd'
[IO.File]::WriteAllText($batch,("@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n"+($commands -join "`r`n")+"`r`n"),[Text.UTF8Encoding]::new($false))
. (Join-Path $PSScriptRoot '../internal/cuda_timing.ps1')
Enable-EcmNvccBatchTiming $batch $Build
& cmd.exe /d /c $batch 2>&1|Out-File (Join-Path $Build 'build.log') -Encoding utf8
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 30;throw 'Stage1 diagnostic adapter build failed'}
Assert-EcmHashes $hashes (Get-EcmSourceHashes $repo $deps) 'Host-wait source changed'
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
$objectHashes=[ordered]@{};foreach($obj in $objects){$objectHashes[$obj]=(Get-FileHash $obj).Hash}
[ordered]@{diagnostic_only=$true;architecture=$Arch;tier=$Tier;cuda=$cuda.VersionText;compiler=$vc.Version;
    sources=$hashes;objects=$objectHashes;executable_sha256=(Get-FileHash $exe).Hash;
    gmp_sha256=(Get-FileHash (Join-Path $Build 'gmp-10.dll')).Hash}|ConvertTo-Json -Depth 6|
    Set-Content (Join-Path $Build 'manifest.json') -Encoding utf8
Write-Host "Diagnostic module probe: $exe"
