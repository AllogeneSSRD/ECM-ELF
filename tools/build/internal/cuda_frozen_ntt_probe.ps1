# Preserve frozen source names/hashes. Legacy engines are copied byte-for-byte
# into an explicit header adapter; no source slicing or frozen manifest edits.
function Invoke-EcmFrozenNttProbeBuild {
    param([string]$Repo,[string]$SourceRoot,[string]$Build,[string]$Arch,
          [string]$Source,[string]$Name,[string[]]$Defines=@(),$AllowedSources=$null)
    . (Join-Path $PSScriptRoot 'cuda_toolchain.ps1')
    . (Join-Path $PSScriptRoot 'cuda_build_state.ps1')
    $cuda=Resolve-EcmCudaToolkit -Archs $Arch;$vc=Resolve-EcmVcEnvironment $cuda
    $SourceRoot=Resolve-EcmSourcePath $Repo $SourceRoot;$Build=Resolve-EcmSourcePath $Repo $Build
    if(Test-Path (Join-Path $Build 'manifest.json')){throw 'Use a fresh build directory'}
    $legacy=Test-Path (Join-Path $SourceRoot 'tools/bench/ntt_poly_probe.cu')
    $engine=if($legacy){'tools/bench/ntt_poly_probe.cu'}else{'tools/experiments/cuda/ntt_variants/engine.cuh'}
    $deps=Get-EcmSourceClosure $SourceRoot @($engine) @('include')
    $hashes=Get-EcmSourceHashes $SourceRoot $deps
    if($AllowedSources) {
        foreach($dep in $deps){if($AllowedSources.$dep -ne $hashes[$dep]){throw "Frozen dependency absent or stale in manifest: $dep"}}
    }
    $snapshot=Join-Path $Build 'sources';$generated=[ordered]@{}
    foreach($dep in $deps) {
        if([IO.Path]::IsPathRooted($dep)){throw 'Frozen project closure must use relative paths'}
        $dest=Join-Path $snapshot $dep;New-Item -ItemType Directory -Force (Split-Path -Parent $dest)|Out-Null
        Copy-Item -LiteralPath (Join-Path $SourceRoot $dep) -Destination $dest
        $generated['sources/'+$dep]=(Get-FileHash $dest).Hash
        if($legacy -and $dep -like 'tools/bench/*') {
            $leaf=if($dep -eq $engine){'engine.cuh'}else{Split-Path -Leaf $dep}
            $relative='tools/experiments/cuda/ntt_variants/'+$leaf
            $alias=Join-Path $snapshot $relative
            New-Item -ItemType Directory -Force (Split-Path -Parent $alias)|Out-Null
            Copy-Item -LiteralPath (Join-Path $SourceRoot $dep) -Destination $alias
            $generated['sources/'+$relative]=(Get-FileHash $alias).Hash
        }
    }
    if($legacy){$Defines+=@('NTT_POLY_PROBE_NO_MAIN=1')}
    $own=@($Source,'tools/build/internal/cuda_frozen_ntt_probe.ps1','tools/build/internal/cuda_build_state.ps1','tools/build/internal/cuda_toolchain.ps1')
    $local=Get-EcmSourceHashes $Repo $own
    $src=Join-Path $snapshot $Source;New-Item -ItemType Directory -Force (Split-Path -Parent $src)|Out-Null
    Copy-Item (Join-Path $Repo $Source) $src;$generated['sources/'+$Source]=(Get-FileHash $src).Hash
    $exe=Join-Path $Build ($Name+'.exe');$gmp=Join-Path $Repo 'third_party/gmp-zen3/dist'
    $command='"'+$cuda.Nvcc+'" -ccbin "'+$vc.Compiler+'" -std=c++17 -O3 -arch='+$Arch+' '+
        (($Defines|ForEach-Object {'-D'+$_}) -join ' ')+' -Xcompiler /utf-8 -Xcompiler /wd4819 '+
        '-I "'+$snapshot+'/include" -I "'+$gmp+'/include" "'+$src+'" -L "'+$gmp+'/lib" -lgmp -o "'+$exe+'"'
    $batch=Join-Path $Build 'compile.cmd'
    [IO.File]::WriteAllText($batch,"@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$command`r`n",[Text.UTF8Encoding]::new($false))
    . (Join-Path $PSScriptRoot 'cuda_timing.ps1')
    Enable-EcmNvccBatchTiming $batch $Build
    & cmd.exe /d /c $batch 2>&1|Out-File (Join-Path $Build 'build.log') -Encoding utf8
    if($LASTEXITCODE -ne 0){Get-Content (Join-Path $Build 'build.log') -Tail 30;throw 'Frozen NTT consumer failed'}
    Assert-EcmHashes $hashes (Get-EcmSourceHashes $SourceRoot $deps) 'Frozen source changed'
    Assert-EcmHashes $local (Get-EcmSourceHashes $Repo $own) 'Local consumer changed'
    Assert-EcmHashes $generated (Get-EcmSourceHashes $Build $generated.Keys) 'Frozen adapter changed'
    Copy-Item (Join-Path $gmp 'bin/gmp-10.dll') $Build -Force
    return [ordered]@{exe=$exe;sha256=(Get-FileHash $exe).Hash.ToLower();source_root=$SourceRoot;
        sources=$hashes;local_sources=$local;generated_sources=$generated;architecture=$Arch;legacy_header_adapter=$legacy}
}
