# Source/object identity shared by Stage1 release, splitting and SkipBuild.
. (Join-Path $PSScriptRoot 'cuda_build_state.ps1')

function Get-Stage1CudaObjects([string]$BuildDir) {
    $objects = @(Get-EcmTargetCompileEntries $BuildDir 'ecm_cuda' |
        Where-Object { $_.file -match '\.cu$' } | ForEach-Object {
            Get-EcmSourceKey $BuildDir (Get-EcmObjectPath $_ $BuildDir)
        } | Sort-Object -Unique)
    if (-not $objects.Count) { throw 'No ecm_cuda CUDA objects in compile_commands.json' }
    return $objects
}

function Get-Stage1ReleaseSources([string]$Repo, [bool]$EnablePrac = $false, [string]$BuildDir = '') {
    $inventory = Get-EcmCudaInventory $Repo
    $sources = @($inventory.groups.STAGE1_COMMON) + @($inventory.groups.STAGE1_CUDA)
    if ($EnablePrac) { $sources += @($inventory.groups.STAGE1_PRAC) }
    $buildInputs = @()
    if ($BuildDir) {
        foreach ($name in @('compile_commands.json','cuda_target_graph.json')) {
            $path = Join-Path $BuildDir $name
            if (-not (Test-Path -LiteralPath $path)) { throw "Stage1 target metadata missing; reconfigure: $name" }
            $buildInputs += Get-EcmSourceKey $Repo $path
        }
        # Linked host modules and generated identity sources are build inputs
        # too, even when they are no longer direct sources of ecm_cuda.
        $sources += @(Get-EcmTargetCompileEntries $BuildDir ecm_cuda | ForEach-Object { $_.file })
    }
    $dirs = @('include','src','src/core','src/cpu','src/cuda/common','kernels/cuda',
        'third_party/cgbn/include','third_party/cgbn/include/cgbn')
    $paths = @(Get-EcmSourceClosure $Repo $sources $dirs -Production) + @($inventory.build_inputs) + $buildInputs
    $paths += @('tools/build/build_stage1_local.ps1','tools/build/build_stage1_dev.ps1',
        'tools/build/build_stage1_release.ps1','tools/build/internal/stage1_release_sources.ps1',
        'tools/build/internal/parallel_nvcc.ps1','tools/build/internal/build_stage1_release_group.ps1',
        'tools/build/internal/split_stage1_release.ps1','tools/build/internal/package_stage1.ps1',
        'tools/build/internal/package_archive.ps1','tools/build/internal/release_payload.ps1')
    return Get-EcmSourceHashes $Repo $paths
}

function Assert-Stage1ReleaseSources($Expected, $Actual) {
    Assert-EcmHashes $Expected $Actual 'Stage1 source changed; rebuild before packaging'
}

function Assert-Stage1BuildIdentity([string]$Repo, [string]$BuildDir, $Manifest) {
    if ($Manifest.identity_schema -ne 1 -or -not $Manifest.compile_setup) { throw 'Stage1 build identity missing; rebuild old manifests' }
    Assert-Stage1ReleaseSources $Manifest.source_hashes (Get-Stage1ReleaseSources $Repo -EnablePrac:$Manifest.enable_prac -BuildDir $BuildDir)
    $objects = @(Get-Stage1CudaObjects $BuildDir)
    if (@($Manifest.objects.PSObject.Properties).Count -ne $objects.Count) { throw 'Stage1 CUDA object set changed' }
    foreach ($object in $objects) {
        if ($Manifest.objects.$object -ne (Get-FileHash -LiteralPath (Join-Path $BuildDir $object)).Hash) { throw "Stage1 CUDA object changed: $object" }
    }
    foreach ($entry in @(Get-EcmTargetCompileEntries $BuildDir ecm_cuda | Where-Object { $_.file -match '\.cu$' })) {
        $state = Get-EcmCompileState $Repo $entry $Manifest.compile_setup -Production
        if (-not (Test-EcmCompileState $state)) { throw "Stage1 CUDA dependency/command changed: $($entry.file)" }
    }
    if ($Manifest.executable_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $BuildDir 'ecm_cuda.exe')).Hash) { throw 'Stage1 executable changed' }
}
