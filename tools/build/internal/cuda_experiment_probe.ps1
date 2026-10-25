# Complete experimental targets with content identities; no source extraction.
function Invoke-EcmExperimentBuild {
    param([string]$Repo,[string]$Build,[string]$Arch,[string[]]$Sources,[string]$Name,
          [string[]]$Defines=@(),[switch]$Rebuild)
    . (Join-Path $PSScriptRoot 'cuda_toolchain.ps1')
    . (Join-Path $PSScriptRoot 'cuda_build_state.ps1')
    $cuda=Resolve-EcmCudaToolkit -Archs $Arch
    $vc=Resolve-EcmVcEnvironment $cuda
    $Build=Resolve-EcmSourcePath $Repo $Build
    $gmp=Resolve-EcmSourcePath $Repo 'third_party/gmp-zen3/dist'
    $dirs=@('include','third_party/cgbn/include','third_party/cgbn/include/cgbn',(Join-Path $gmp 'include'))
    $deps=@(Get-EcmSourceClosure $Repo $Sources $dirs)+@(
        'tools/build/internal/cuda_experiment_probe.ps1','tools/build/internal/cuda_build_state.ps1',
        'tools/build/internal/cuda_toolchain.ps1','tools/build/internal/build_directory.ps1')
    $hashes=Get-EcmSourceHashes $Repo $deps
    $names=Get-EcmObjectNames $Sources
    . (Join-Path $PSScriptRoot 'build_directory.ps1')
    Initialize-EcmBuildDirectory $Build
    $objects=@();$commands=@()
    foreach($source in $Sources) {
        $obj=Join-Path $Build ($names[$source]+'.obj');$objects+=$obj
        $commands+=('"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" -std=c++17 -O3 -arch='+$Arch+
            ' --pre-include=stdint.h -Xcompiler /MD -Xcompiler /utf-8 -Xcompiler /wd4819 '+
            (($dirs|ForEach-Object {'-I "'+(Resolve-EcmSourcePath $Repo $_)+'"'}) -join ' ')+' '+
            (($Defines|ForEach-Object {'-D'+$_}) -join ' ')+' -c "'+$source+'" -o "'+$obj+'"')
        $commands+='if errorlevel 1 exit /b 1'
    }
    $exe=Join-Path $Build ($Name+'.exe')
    $commands+=('"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" -Xcompiler /MD '+(($objects|ForEach-Object {'"'+$_+'"'}) -join ' ')+
        ' -L "'+$gmp+'/lib" -lgmp -o "'+$exe+'"')
    $commands+='if errorlevel 1 exit /b 1'
    $batch=Join-Path $Build 'compile.cmd';$log=Join-Path $Build 'build.log'
    $body="@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n"+($commands -join "`r`n")+"`r`n"
    $identity=[ordered]@{sources=$hashes;commands=$body;cuda=$cuda.VersionText;compiler=$vc.Version;
        gmp=(Get-FileHash (Join-Path $gmp 'bin/gmp-10.dll')).Hash}
    $manifest=Join-Path $Build 'manifest.json'
    $reused=$false
    if(-not $Rebuild -and (Test-Path $manifest) -and (Test-Path $exe)) {
        $previous=Get-Content $manifest -Raw|ConvertFrom-Json
        if(($previous.identity|ConvertTo-Json -Depth 10 -Compress) -eq ($identity|ConvertTo-Json -Depth 10 -Compress)) {
            $reused=$previous.sha256 -eq (Get-FileHash $exe).Hash
            foreach($obj in $objects) {
                if(-not (Test-Path $obj) -or $previous.objects.$obj -ne (Get-FileHash $obj).Hash){$reused=$false}
            }
        }
    }
    $watch=[Diagnostics.Stopwatch]::StartNew()
    if(-not $reused) {
        [IO.File]::WriteAllText($batch,$body,[Text.UTF8Encoding]::new($false))
        . (Join-Path $PSScriptRoot 'cuda_timing.ps1')
        Enable-EcmNvccBatchTiming $batch $Build
        & cmd.exe /d /c $batch 2>&1|Out-File $log -Encoding utf8
        if($LASTEXITCODE -ne 0){Get-Content $log -Tail 35;throw "Experimental target failed: $Name"}
    }
    Assert-EcmHashes $hashes (Get-EcmSourceHashes $Repo $deps) 'Experiment source changed during build'
    Copy-Item (Join-Path $gmp 'bin/gmp-10.dll') $Build -Force
    $objectHashes=[ordered]@{};foreach($obj in $objects){$objectHashes[$obj]=(Get-FileHash $obj).Hash}
    [ordered]@{experimental=$true;production_runtime=$false;exe=$exe;sha256=(Get-FileHash $exe).Hash;
        architecture=$Arch;identity=$identity;sources=$hashes;generated_sources=[ordered]@{};
        objects=$objectHashes;reused=$reused;build_seconds=$watch.Elapsed.TotalSeconds}|
        ConvertTo-Json -Depth 12|Set-Content $manifest -Encoding utf8
    Write-Host "built experimental $exe (reused=$reused)"
}
