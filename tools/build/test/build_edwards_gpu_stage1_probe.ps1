#Requires -Version 5.1
param([string]$Build='build_cuda_cmake/edwards_gpu_stage1_probe',
      [ValidatePattern('^sm_[0-9]+$')][string]$Arch='sm_89',
      [int]$Jobs=8,
      [string]$Tiers='',
      [string]$WorkerBuild='',
      [switch]$Incremental)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/cuda_toolchain.ps1')
$cuda=Resolve-EcmCudaToolkit -Archs $Arch
$vc=Resolve-EcmVcEnvironment $cuda
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
$objDir=Join-Path $Build 'CMakeFiles/edwards_gpu_stage1_probe.dir'
New-Item -ItemType Directory -Force $objDir | Out-Null
. (Join-Path $PSScriptRoot '../internal/stage1_release_sources.ps1')
$inventory=Get-EcmCudaInventory $repo
$workerSources=@($inventory.groups.STAGE1_EDWARDS)
$sources=$workerSources+
    @('tools/test/edwards_gpu_stage1_probe.cpp','src/cpu/ecm_edwards_cpu.cpp')
$deps=@(Get-EcmSourceClosure $repo $sources @('include','src/core','src/cpu','third_party/cgbn/include'))+
    @('tools/build/test/build_edwards_gpu_stage1_probe.ps1')
$hashes=[ordered]@{}
foreach($source in ($sources+$deps|Select-Object -Unique)){$hashes[$source]=(Get-FileHash -LiteralPath $source).Hash}
$worker=$null
if($WorkerBuild){
    $WorkerBuild=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkerBuild)
    $local=Get-Content (Join-Path $WorkerBuild 'stage1_local_manifest.json') -Raw|ConvertFrom-Json
    if("sm_$($local.architecture)" -ne $Arch){throw 'worker architecture mismatch'}
    if($local.tiers -ne $Tiers){throw 'worker tier selection mismatch'}
    Assert-Stage1BuildIdentity $repo $WorkerBuild $local
    $worker=[pscustomobject]@{objects=[pscustomobject]@{}}
    foreach($entry in @(Get-EcmTargetCompileEntries $WorkerBuild ecm_cuda)){
        if((Get-EcmSourceKey $repo $entry.file) -notin $workerSources){continue}
        $path=Get-EcmObjectPath $entry $WorkerBuild
        $worker.objects|Add-Member -NotePropertyName $entry.file -NotePropertyValue ([pscustomobject]@{path=$path;sha256=(Get-FileHash $path).Hash})
    }
}
$previous=$null;$manifestPath=Join-Path $Build 'manifest.json'
if($Incremental -and (Test-Path $manifestPath)){$previous=Get-Content $manifestPath -Raw|ConvertFrom-Json}
$reuse=$previous -and $previous.arch -eq $Arch -and $previous.tiers -eq $Tiers
if($reuse){foreach($key in $hashes.Keys){if($previous.sources.$key -ne $hashes[$key]){$reuse=$false}}}
$objects=@();$objectMap=[ordered]@{};$commands=@()
foreach($source in $sources){
    $stem=[IO.Path]::GetFileNameWithoutExtension($source)
    $obj=Join-Path $objDir ($stem+'.obj')
    if($worker -and $source -in $workerSources){
        $absolute=(Join-Path $repo $source).Replace('\','/')
        $entry=@($worker.objects.PSObject.Properties|Where-Object{$_.Name.Replace('\','/') -eq $absolute})
        if($entry.Count -ne 1){throw "worker object absent: $source"}
        $obj=$entry[0].Value.path
        if($entry[0].Value.sha256 -ne (Get-FileHash -LiteralPath $obj).Hash){throw "worker object changed: $source"}
        Write-Host "Reused verified worker object: $source"
    } elseif(-not ($reuse -and $previous.objects.$source -and (Test-Path $obj) -and
        $previous.objects.$source -eq (Get-FileHash -LiteralPath $obj).Hash)) {
        $command='"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" -std=c++17 -O2 -arch='+$Arch+
            ' --pre-include=stdint.h -Xcompiler /MD -Xcompiler /utf-8 -Xcompiler /wd4819 -I include -I third_party/cgbn/include -I third_party/gmp-zen3/dist/include -I src/core -I src/cpu -I src/cuda/stage1 -c "'+$source+'" -o "'+$obj+'"'
        if($Tiers){
            $definitions=' -DECM_TIERS_RESTRICTED'
            foreach($tier in $Tiers.Split(',')){
                if($tier -notmatch '^\d+$'){throw 'Tiers must be comma-separated numeric containers'}
                $definitions+=' -DECM_TIER_'+$tier
            }
            $command+=$definitions
        }
        $commands += [ordered]@{directory=$repo;file=(Join-Path $repo $source);command=$command}
    } else {Write-Host "Reused unchanged object: $source"}
    $objects+=$obj;$objectMap[$source]=$obj
}
if($commands.Count){
    ConvertTo-Json -InputObject @($commands) -Depth 4|Set-Content (Join-Path $Build 'compile_commands.json') -Encoding utf8
    & (Join-Path $PSScriptRoot '../internal/parallel_nvcc.ps1') -BuildDir $Build -Target edwards_gpu_stage1_probe `
        -Jobs $Jobs -SourceRegex '\.(cu|cpp)$' -NoBuild -VcVars $vc.VcVars -VcVersion $vc.Version
    if($LASTEXITCODE -ne 0){throw 'parallel probe compile failed'}
}
$objArgs=($objects|ForEach-Object{'"'+$_+'"'}) -join ' '
$exe=Join-Path $Build 'edwards_gpu_stage1_probe.exe'
$command='"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" -arch='+$Arch+' -Xcompiler /MD '+$objArgs+' -L third_party/gmp-zen3/dist/lib -lgmp -o "'+$exe+'"'
$cmdFile=Join-Path $Build 'link.cmd'
[IO.File]::WriteAllText($cmdFile,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
& cmd.exe /d /c $cmdFile 2>&1 | Out-File (Join-Path $Build 'link.log') -Encoding utf8
if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'link.log') -Tail 30;throw 'link failed'}
Copy-Item third_party/gmp-zen3/dist/bin/gmp-10.dll $Build -Force
foreach($key in $hashes.Keys){if($hashes[$key] -ne (Get-FileHash -LiteralPath $key).Hash){throw "source changed during probe build: $key"}}
$objectHashes=[ordered]@{}
foreach($source in $sources){$objectHashes[$source]=(Get-FileHash -LiteralPath $objectMap[$source]).Hash}
[ordered]@{exe=$exe;sha256=(Get-FileHash $exe).Hash;arch=$Arch;tiers=$Tiers;worker_build=$WorkerBuild;objects=$objectHashes;
    cuda_object_sha256=$objectHashes['src/cuda/stage1/edwards_runtime.cu'];sources=$hashes
}|ConvertTo-Json -Depth 5|Set-Content $manifestPath -Encoding utf8
Write-Host "Built $exe"
