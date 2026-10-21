# Production NTT consumers compile/link the explicit inventory, never a copied
# implementation header. Per-thread experiments build every unit with that ABI.
function Invoke-EcmNttProbeBuild {
    param([string]$Repo,[string]$Build,[string]$Arch,[string]$Source,[string]$Name,
          [ValidateSet('legacy','per-thread')][string]$DefaultStream='legacy',
          [string]$WorkerBuild='',[string]$Gmp='third_party/gmp-zen3/dist',
          [ValidateSet('STAGE2_NTT','STAGE2_ARITHMETIC')][string]$ModuleGroup='STAGE2_NTT',
          [string[]]$AdditionalProductionSources=@(),[string[]]$ExtraSources=@(),[string[]]$Defines=@())
    . (Join-Path $PSScriptRoot 'cuda_toolchain.ps1')
    . (Join-Path $PSScriptRoot 'cuda_build_state.ps1')
    $cuda=Resolve-EcmCudaToolkit -Archs $Arch
    $vc=Resolve-EcmVcEnvironment $cuda
    $Build=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
    $Gmp=Resolve-EcmSourcePath $Repo $Gmp
    $inventory=Get-EcmCudaInventory $Repo
    $moduleSources=@($inventory.groups.$ModuleGroup)+$AdditionalProductionSources
    $sources=@($Source)+$ExtraSources+$moduleSources
    $dirs=@((Join-Path $Gmp 'include'),'include','src','src/ecm/backend/cuda')
    $deps=@(Get-EcmSourceClosure $Repo $sources $dirs)
    foreach($dep in $deps) {
        if($dep -match '^(src|kernels|include|tools)/' -and [IO.Path]::GetExtension($dep) -in @('.cu','.cuh','.cpp','.h')) {
            $body=[IO.File]::ReadAllText((Resolve-EcmSourcePath $Repo $dep))
            if($body -match '(?m)^\s*#\s*include\s*"[^"\r\n]+\.(cu|cpp|inl)"') {
                throw "Probe includes an implementation instead of linking a module: $dep"
            }
        }
    }
    $deps+=@('tools/build/internal/cuda_ntt_probe.ps1','tools/build/internal/cuda_sources.cmake',
             'tools/build/internal/cuda_build_state.ps1','tools/build/internal/cuda_toolchain.ps1')
    $hashes=Get-EcmSourceHashes $Repo $deps
    New-Item -ItemType Directory -Force $Build | Out-Null
    $objects=@();$objectHashes=[ordered]@{};$compile=@();$watch=[Diagnostics.Stopwatch]::StartNew()
    if($WorkerBuild) {
        if($DefaultStream -ne 'legacy'){throw 'Production objects use the legacy default stream'}
        $WorkerBuild=Resolve-EcmSourcePath $Repo $WorkerBuild
        $worker=Get-Content (Join-Path $WorkerBuild 'build_manifest.json') -Raw | ConvertFrom-Json
        if($worker.engine -ne 'production' -or $worker.architecture -ne $Arch -or $worker.gl_fixed_mode -ne 3 -or
           $worker.add_sub_mask -ne 1 -or $worker.outer_unroll_u -ne 0 -or $worker.gmp_root -ne $Gmp){throw 'Worker ABI differs'}
        $set=Get-EcmStage2SourceSet $Repo $inventory production $dirs
        Assert-EcmBuildUnits $Repo $WorkerBuild $worker $set.Sources $dirs -Production
        foreach($src in $moduleSources) {
            $obj=$worker.compile_units.$src.entry.output
            if(-not $obj){throw "Missing production module object: $src"}
            $objects+=('"'+$obj+'"');$objectHashes[$obj]=(Get-FileHash $obj).Hash
        }
        $sources=@($Source)+$ExtraSources
    }
    $names=Get-EcmObjectNames $sources
    foreach($src in $sources) {
        $obj=Join-Path $Build ($names[$src]+'.obj')
        $compile+=('"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" -std=c++17 -O3 --default-stream '+$DefaultStream+
            ' -arch='+$Arch+' -DNTT_GL_FIXED_MODE=3 -DNTT_GL_ADD_SUB_MASK=1 -DNTT_OUTER_UNROLL_U=0'+
            ' '+(($Defines|ForEach-Object {'-D'+$_}) -join ' ')+
            ' -I "'+$Repo+'/include" -I "'+$Gmp+'/include" -Xcompiler /utf-8 -Xcompiler /wd4819 -c "'+$src+'" -o "'+$obj+'"')
        $compile+='if errorlevel 1 exit /b 1';$objects+=('"'+$obj+'"')
    }
    $exe=Join-Path $Build ($Name+'.exe')
    $compile+=('"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" '+($objects -join ' ')+' -L "'+$Gmp+'/lib" -lgmp -o "'+$exe+'"')
    $batch=Join-Path $Build 'compile.cmd';$log=Join-Path $Build 'build.log'
    [IO.File]::WriteAllText($batch,("@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n"+($compile -join "`r`n")+"`r`n"),[Text.UTF8Encoding]::new($false))
    & cmd.exe /d /c $batch 2>&1 | Out-File $log -Encoding utf8
    if($LASTEXITCODE -ne 0){Get-Content $log -Tail 40;throw "NTT consumer build failed: $Source"}
    $compiledObjects=[ordered]@{}
    foreach($src in $sources) {
        $obj=Join-Path $Build ($names[$src]+'.obj')
        $compiledObjects[$obj]=(Get-FileHash $obj).Hash
    }
    foreach($dep in $hashes.Keys){if((Get-FileHash (Resolve-EcmSourcePath $Repo $dep)).Hash -ne $hashes[$dep]){throw "Source changed: $dep"}}
    foreach($obj in $objectHashes.Keys){if((Get-FileHash $obj).Hash -ne $objectHashes[$obj]){throw "Production object changed: $obj"}}
    Copy-Item (Join-Path $Gmp 'bin/gmp-10.dll') $Build -Force
    $watch.Stop()
    [ordered]@{sha256=(Get-FileHash $exe).Hash;architecture=$Arch;default_stream=$DefaultStream;
        gl_fixed_mode=3;production_runtime=($ModuleGroup -eq 'STAGE2_NTT');production_module_group=$ModuleGroup;
        build_seconds=$watch.Elapsed.TotalSeconds;toolkit=$cuda.VersionText;msvc=$vc.Version;
        gmp_dll_sha256=(Get-FileHash (Join-Path $Build 'gmp-10.dll')).Hash;sources=$hashes;
        production_sources=$moduleSources;extra_sources=$ExtraSources;defines=$Defines;
        production_objects=$objectHashes;compiled_objects=$compiledObjects;worker=$WorkerBuild} |
        ConvertTo-Json -Depth 6 | Set-Content (Join-Path $Build 'manifest.json') -Encoding utf8
    Write-Host "built $exe"
}
