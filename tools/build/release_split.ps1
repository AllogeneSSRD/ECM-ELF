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

    The multi-arch build must be configured like this (docs/ECM_CGBN_OPTIMIZATION.md 8.9):

        cmake -S . -B build_cuda_release -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release `
              -DECM_CUDA_FULL_BUILD=ON -DECM_CUDA_ARCHITECTURES="75;86;89;90;100;120" `
              -DECM_CUDA_PTX_ARCH=75 -DECM_CUDA_COMPRESS=OFF -DECM_TPB=128 -DECM_MAX_ROTATION=1 ...
        powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\parallel_nvcc.ps1 -BuildDir build_cuda_release

    Keep -DECM_CUDA_COMPRESS=OFF here: nvprune rewrites the fatbins, and uncompressed fatbins make
    that rewrite predictable.  The single-arch local test build (build_cuda_cmake) is the opposite:
    compression ON, no PTX, one arch.

.PARAMETER BuildDir   Multi-arch build directory that already contains the CUDA objects.
.PARAMETER OutDir     Where the per-arch exes go (plus a backup of the unpruned objects).
.PARAMETER Archs      Architectures to emit, e.g. "75,86,89,90,100,120".
.PARAMETER PtxArch    Architecture that also keeps its PTX (JIT fallback for newer GPUs), default 75.
#>
param(
    [string]$BuildDir = "build_cuda_release",
    [string]$OutDir = "dist\cuda",
    [string]$Archs = "75,86,89,90,100,120",
    [string]$PtxArch = "75",
    [string]$VcVars = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not [System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
if (-not [System.IO.Path]::IsPathRooted($OutDir)) { $OutDir = Join-Path $repo $OutDir }
if (-not (Test-Path $BuildDir)) { throw "build dir not found: $BuildDir" }

$cuda = $env:CUDA_PATH
if (-not $cuda) { $cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3" }
$nvprune = Join-Path $cuda "bin\nvprune.exe"
$cuobjdump = Join-Path $cuda "bin\cuobjdump.exe"
if (-not (Test-Path $nvprune)) { throw "nvprune.exe not found (expected $nvprune)" }
if (-not $VcVars) {
    $cand = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending
    if (-not $cand) { throw "vcvars64.bat not found; pass -VcVars" }
    $VcVars = $cand[0].FullName
}

$objDir = Join-Path $BuildDir "CMakeFiles\ecm_cuda.dir\kernels\cuda"
$cuObjs = @(Get-ChildItem $objDir -Filter *.obj -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name)
if (-not $cuObjs) { throw "no .cu.obj in $objDir -- build the target once first" }

New-Item -ItemType Directory -Force $OutDir | Out-Null
$backup = Join-Path $OutDir "_obj_backup"
New-Item -ItemType Directory -Force $backup | Out-Null
Copy-Item (Join-Path $objDir *.obj) $backup -Force

Write-Host ("release split: archs {0}, PTX kept for {1}; {2} CUDA objects" -f $Archs, $PtxArch, $cuObjs.Count)
try {
    foreach ($arch in ($Archs -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
        $dstDir = Join-Path $OutDir "obj_sm$arch"
        New-Item -ItemType Directory -Force $dstDir | Out-Null

        # 1) prune every CUDA object down to this architecture (+ PTX for the fallback arch)
        $gencode = "arch=compute_$arch,code=sm_$arch"
        if ($arch -eq $PtxArch) { $gencode = "arch=compute_$arch,code=[sm_$arch,compute_$arch]" }
        foreach ($o in $cuObjs) {
            $src = Join-Path $objDir $o
            $dst = Join-Path $dstDir $o
            & $nvprune -gencode $gencode $src -o $dst 2>&1 |
                Where-Object { $_ -match 'error|fatal' } | ForEach-Object { Write-Host "    $_" }
            if (-not (Test-Path $dst)) { throw "nvprune produced nothing for $o (arch $arch)" }
            Copy-Item $dst $src -Force
        }

        # 2) relink with the pruned objects in place, then take the exe
        & cmd.exe /c "call `"$VcVars`" >nul 2>&1 && cmake --build `"$BuildDir`" --target ecm_cuda" | Out-Null
        $exe = Join-Path $OutDir "ecm_cuda_sm$arch.exe"
        Copy-Item (Join-Path $BuildDir "ecm_cuda.exe") $exe -Force

        $cubins = (& $cuobjdump --list-elf $exe 2>&1 | Select-String 'cubin' |
                   ForEach-Object { ($_.Line -split '\.sm_')[-1] -replace '\.cubin', '' } |
                   Sort-Object -Unique) -join ','
        $ptx = ((& $cuobjdump --list-ptx $exe 2>&1 | Select-String '\.ptx').Count)
        $mb = (Get-Item $exe).Length / 1MB
        Write-Host ("  sm_{0,-4} -> {1}  {2,6:N1} MB  cubins=sm_{3}  ptx entries={4}" -f `
                    $arch, (Split-Path $exe -Leaf), $mb, $cubins, $ptx)
    }
} finally {
    Copy-Item (Join-Path $backup *.obj) $objDir -Force
    Write-Host "  (unpruned objects restored in $objDir)"
}
