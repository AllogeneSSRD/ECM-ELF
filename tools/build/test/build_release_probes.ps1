# Independent GMP probes linked to the exact selected release objects.
param([ValidateSet('cu11.8','cu13.3')][string]$Profile='cu13.3',
    [ValidateSet('stage1','stage2')][string]$Stage='stage2',
    [string]$BuildRoot='build_cuda_release',[Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot));Set-Location $repo
. (Join-Path $PSScriptRoot '../internal/stage1_release_sources.ps1')
. (Join-Path $PSScriptRoot '../internal/release_profiles.ps1')
. (Join-Path $PSScriptRoot '../internal/cuda_timing.ps1')
$record=Get-Content (Join-Path $BuildRoot "components/$Profile.$Stage.json") -Raw | ConvertFrom-Json
$manifest=Get-Content -LiteralPath $record.manifest -Raw | ConvertFrom-Json
$target=Get-EcmReleaseTarget $Profile
$tools=Resolve-EcmReleaseToolchain $target $manifest.cuda_root
if($tools.Vc.Compiler -ine $manifest.host_compiler){throw 'Probe compiler differs from release'}
$Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
. (Join-Path $PSScriptRoot '../internal/build_directory.ps1')
Initialize-EcmBuildDirectory $Build
$inventory=Get-EcmCudaInventory $repo
$objects=@();$sources=@()
if($Stage -eq 'stage1') {
    Assert-Stage1BuildIdentity $repo $record.build $manifest
    $gmp=$manifest.gmp
    $wanted=@($inventory.groups.STAGE1_EDWARDS|Where-Object {$_ -notmatch '/instances/'} )+@('src/ecm/stage1/scalar.cpp')
    foreach($entry in @(Get-EcmTargetCompileEntries $record.build ecm_cuda)) {
        $key=Get-EcmSourceKey $repo $entry.file
        if($key -in $wanted -or $entry.file.Replace('\','/') -match '/cuda_modules/stage1/edwards_\d+\.cu$') {
            $objects+=Get-EcmObjectPath $entry $record.build
        }
    }
    if(@($objects|Where-Object {$_ -match 'edwards_\d+\.cu\.obj$'}).Count -ne 32){throw 'Probe requires the complete32-tier release'}
    $sources=@('tools/test/edwards_gpu_stage1_probe.cpp','src/ecm/edwards_stage1/cpu/ecm_edwards_cpu.cpp')
    $name='edwards_gpu_stage1_probe.exe';$crt='/MD'
} else {
    $gmp=$manifest.gmp_root
    $dirs=@((Join-Path $gmp 'include'),'include','src','src/ecm/backend/cuda')
    $set=Get-EcmStage2SourceSet $repo $inventory production $dirs $manifest.module_directory
    Assert-EcmBuildUnits $repo $record.build $manifest $set.Sources $dirs -Production
    foreach($property in $manifest.compile_units.PSObject.Properties) {
        $source=$property.Name.Replace('\','/')
        if($manifest.module_directory) {
            if($source -match '/arithmetic_[^/]+\.cu$|/stage2/constants\.cpp$'){$objects+=$property.Value.entry.output}
        } elseif($source -eq 'src/ecm/backend/cuda/stage2/device/arithmetic_kernels.cu') {$objects+=$property.Value.entry.output}
    }
    if(($manifest.module_directory -and $objects.Count -lt 2) -or (-not $manifest.module_directory -and $objects.Count -ne 1)){throw 'Selected arithmetic object set missing'}
    $sources=@('tools/test/cuda_stage2_arithmetic_probe.cpp');$name='probe.exe';$crt='/MT'
}
$frozen=[ordered]@{}
foreach($object in $objects){$frozen[$object]=(Get-FileHash -LiteralPath $object).Hash}
$inputHashes=Get-EcmSourceHashes $repo (@(Get-EcmSourceClosure $repo $sources @('include','src',(Join-Path $gmp 'include')))+@('tools/build/test/build_release_probes.ps1'))
$commands=@()
for($i=0;$i -lt $sources.Count;$i++) {
    $obj=Join-Path $Build "probe_$i.obj"
    $commands+='"'+$tools.Cuda.Nvcc+'" -ccbin "'+$tools.Vc.Compiler+'" -std=c++17 -O2 -arch=sm_'+$target.Sass+
        ' --pre-include=stdint.h -Xcompiler '+$crt+' -Xcompiler /utf-8 -Xcompiler /wd4819 -I "'+$gmp+'/include" -I "'+$repo+'/include" -I "'+$repo+'/src" -I "'+$repo+'/third_party/cgbn/include" -c "'+$sources[$i]+'" -o "'+$obj+'"'
    $commands+='if errorlevel 1 exit /b 1';$objects+=$obj
}
$exe=Join-Path $Build $name
$response=Join-Path $Build 'objects.rsp'
[IO.File]::WriteAllText($response,(($objects|ForEach-Object{'"'+$_+'"'}) -join ' ')+" -L `"$gmp/lib`" -lgmp -o `"$exe`"",[Text.UTF8Encoding]::new($false))
$commands+='"'+$tools.Cuda.Nvcc+'" -ccbin "'+$tools.Vc.Compiler+'" -Xcompiler '+$crt+' --options-file "'+$response+'"'
$batch=Join-Path $Build 'build.cmd'
[IO.File]::WriteAllText($batch,"@echo off`r`n$($tools.Vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n"+($commands -join "`r`n")+"`r`n",[Text.UTF8Encoding]::new($false))
Enable-EcmNvccBatchTiming $batch $Build
& cmd.exe /d /c $batch 2>&1 | Out-File (Join-Path $Build 'build.log') -Encoding utf8
if($LASTEXITCODE){Get-Content (Join-Path $Build 'build.log') -Tail 25;throw 'Release-object probe build failed'}
foreach($object in $frozen.Keys){if($frozen[$object] -ne (Get-FileHash -LiteralPath $object).Hash){throw "Release object changed: $object"}}
Assert-EcmHashes ($inputHashes|ConvertTo-Json -Depth 3|ConvertFrom-Json) (Get-EcmSourceHashes $repo $inputHashes.Keys) 'Probe source changed'
Copy-Item -LiteralPath (Join-Path $gmp 'bin/gmp-10.dll') -Destination $Build -Force
[ordered]@{profile=$Profile;stage=$Stage;worker_manifest_sha256=(Get-FileHash $record.manifest).Hash;
    exe=$exe;exe_sha256=(Get-FileHash $exe).Hash;production_objects=$frozen;sources=$inputHashes;
    gmp_dll_sha256=(Get-FileHash (Join-Path $Build 'gmp-10.dll')).Hash} |
    ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Build 'manifest.json') -Encoding utf8
Write-Host "Built independent release-object GMP probe: $exe"
