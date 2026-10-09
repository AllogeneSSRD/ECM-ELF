<#
.SYNOPSIS
    Emit one ecm_cuda.exe per GPU architecture from ONE multi-arch build directory.

.DESCRIPTION
    A single exe carrying every architecture is large, and nvprune cannot be applied to it:

        nvprune fatal : Input file '...ecm_cuda.exe' not relocatable.

    nvprune only accepts relocatable inputs (the .cu.obj files, or a .lib), so the release flow
    is: build once for all target architectures, prune the CUDA objects per architecture, then
    let CMake relink the exe with the pruned objects in place (a relink takes seconds).  The
    original objects are restored afterwards, so the build directory stays usable.

    The multi-arch build must be configured like this (docs/performance/STAGE1.md):

        cmake -S . -B build_cuda_release -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release `
              -DECM_CUDA_FULL_BUILD=ON -DECM_CUDA_ARCHITECTURES="75;86;89;120" `
              -DECM_CUDA_PTX_ARCH=75 -DECM_CUDA_COMPRESS=OFF -DECM_TPB=128 -DECM_MAX_ROTATION=1 ...
        powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\internal\parallel_nvcc.ps1 -BuildDir build_cuda_release

    Keep -DECM_CUDA_COMPRESS=OFF here: nvprune rewrites the fatbins, and uncompressed fatbins make
    that rewrite predictable.  The single-arch local test build (build_cuda_cmake) is the opposite:
    compression ON, no PTX, one arch.

.PARAMETER BuildDir   Multi-arch build directory that already contains the CUDA objects.
.PARAMETER OutDir     Where the per-arch exes go. Manifests and scratch stay in BuildDir.
.PARAMETER Archs      Architectures from one toolkit group, e.g. "60,70" or "75,86,89,120".
.PARAMETER PtxArch    Architecture that also keeps PTX (JIT fallback); empty means SASS only.
#>
param(
    [string]$BuildDir = "build_cuda_release",
    [string]$OutDir = "dist\cuda",
    [string]$Archs = "75,86,89,120",
    [string]$PtxArch = "",
    [string]$VcVars = "",
    [string]$CudaRoot = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if (-not [System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
if (-not [System.IO.Path]::IsPathRooted($OutDir)) { $OutDir = Join-Path $repo $OutDir }
if (-not (Test-Path $BuildDir)) { throw "build dir not found: $BuildDir" }

. (Join-Path $PSScriptRoot 'cuda_toolchain.ps1')
$cuda = Resolve-EcmCudaToolkit -Archs $Archs -CudaRoot $CudaRoot
$vc = Resolve-EcmVcEnvironment $cuda -VcVars $VcVars
$nvprune = Join-Path $cuda.Root "bin\nvprune.exe"
$cuobjdump = Join-Path $cuda.Root "bin\cuobjdump.exe"
if (-not (Test-Path $nvprune)) { throw "nvprune.exe not found (expected $nvprune)" }

. (Join-Path $PSScriptRoot 'stage1_release_sources.ps1')
$cuObjs = @(Get-Stage1CudaObjects $BuildDir)
$manifestPath = Join-Path $BuildDir 'stage1_build_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$originalExe = Join-Path $BuildDir 'ecm_cuda.exe'
if ($manifest.architectures -ne $Archs -or $manifest.ptx_architecture -ne $PtxArch -or
    $manifest.cuda_root -ne $cuda.Root -or $manifest.host_compiler -ne $vc.Compiler -or
    $manifest.executable_sha256 -ne (Get-FileHash -LiteralPath $originalExe).Hash) { throw 'Stage1 split/build identity mismatch' }
if (@($manifest.objects.PSObject.Properties).Count -ne $cuObjs.Count) { throw 'Stage1 CUDA object set changed' }
foreach ($o in $cuObjs) {
    if ($manifest.objects.$o -ne (Get-FileHash -LiteralPath (Join-Path $BuildDir $o)).Hash) { throw "Stage1 CUDA object changed: $o" }
}

New-Item -ItemType Directory -Force $OutDir | Out-Null
. (Join-Path $PSScriptRoot 'release_payload.ps1')
Remove-EcmReleaseMetadata $OutDir
$work = Join-Path $BuildDir ('_release_split/' + [Guid]::NewGuid().ToString('N'))
$backup = Join-Path $work "original"
New-Item -ItemType Directory -Force $backup | Out-Null
foreach ($o in $cuObjs) {
    $backupObject = Join-Path $backup $o
    New-Item -ItemType Directory -Path (Split-Path -Parent $backupObject) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $BuildDir $o) -Destination $backupObject
}
Copy-Item -LiteralPath $originalExe -Destination (Join-Path $backup 'ecm_cuda.exe')
$splitHashes = [ordered]@{}

Write-Host ("release split: archs {0}, PTX kept for {1}; {2} CUDA objects" -f $Archs, $(if ($PtxArch) { $PtxArch } else { 'none' }), $cuObjs.Count)
try {
    foreach ($arch in ($Archs -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
        $dstDir = Join-Path $work "obj_sm$arch"
        New-Item -ItemType Directory -Force $dstDir | Out-Null

        # 1) prune every CUDA object down to this architecture (+ PTX for the fallback arch)
        $gencode = "arch=compute_$arch,code=sm_$arch"
        if ($arch -eq $PtxArch) { $gencode = "arch=compute_$arch,code=[sm_$arch,compute_$arch]" }
        foreach ($o in $cuObjs) {
            # Always prune the original multi-architecture object, not the previous split.
            $src = Join-Path $backup $o
            $buildObject = Join-Path $BuildDir $o
            $dst = Join-Path $dstDir $o
            New-Item -ItemType Directory -Path (Split-Path -Parent $dst) -Force | Out-Null
            & $nvprune -gencode $gencode $src -o $dst 2>&1 |
                Where-Object { $_ -match 'error|fatal' } | ForEach-Object { Write-Host "    $_" }
            if ($LASTEXITCODE -ne 0) { throw "nvprune failed: $o (sm_$arch)" }
            if (-not (Test-Path $dst)) { throw "nvprune produced nothing for $o (arch $arch)" }
            Copy-Item -LiteralPath $dst -Destination $buildObject -Force
        }

        # 2) relink with the pruned objects in place, then take the exe
        & cmd.exe /c "$($vc.Setup) && cmake --build `"$BuildDir`" --target ecm_cuda" | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Stage1 relink failed: sm_$arch" }
        $exe = Join-Path $OutDir "ecm_cuda_sm$arch.exe"
        Copy-Item (Join-Path $BuildDir "ecm_cuda.exe") $exe -Force

        $elfOutput = @(& $cuobjdump --list-elf $exe 2>&1)
        if ($LASTEXITCODE -ne 0) { throw "Cannot inspect cubins: sm_$arch" }
        $cubins = @($elfOutput | Select-String '\.sm_([0-9]+)\.cubin' |
                   ForEach-Object { $_.Matches[0].Groups[1].Value } | Sort-Object -Unique)
        if ($cubins.Count -ne 1 -or $cubins[0] -ne $arch) { throw "Split executable has unexpected cubins: sm_$arch ($cubins)" }
        $ptxOutput = @(& $cuobjdump --list-ptx $exe 2>&1)
        if ($LASTEXITCODE -ne 0) { throw "Cannot inspect PTX: sm_$arch" }
        $ptx = @($ptxOutput | Select-String '\.ptx').Count
        if (($arch -eq $PtxArch -and $ptx -eq 0) -or ($arch -ne $PtxArch -and $ptx -ne 0)) { throw "Unexpected PTX policy: sm_$arch" }
        $splitHashes[[IO.Path]::GetFileName($exe)] = (Get-FileHash -LiteralPath $exe).Hash
        $mb = (Get-Item $exe).Length / 1MB
        Write-Host ("  sm_{0,-4} -> {1}  {2,6:N1} MB  cubins=sm_{3}  ptx entries={4}" -f `
                    $arch, (Split-Path $exe -Leaf), $mb, ($cubins -join ','), $ptx)
    }
} finally {
    foreach ($o in $cuObjs) { Copy-Item -LiteralPath (Join-Path $backup $o) -Destination (Join-Path $BuildDir $o) -Force }
    Copy-Item -LiteralPath (Join-Path $backup 'ecm_cuda.exe') -Destination $originalExe -Force
    Write-Host "  (unpruned objects and multi-architecture executable restored)"
}
[ordered]@{ build_manifest_sha256 = (Get-FileHash -LiteralPath $manifestPath).Hash;
    executables = $splitHashes; architectures = $Archs; ptx_architecture = $PtxArch } |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $BuildDir 'split_manifest.json') -Encoding UTF8
