#Requires -Version 5.1
<#
.SYNOPSIS
Build the standalone CUDA Stage2 save/queue executable (MSVC + nvcc + GMP).
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_ecm_cuda_stage2.ps1 -Arch sm_89
#>
param(
    [string]$Build = 'build_cuda_cmake/production_stage2',
    [ValidateSet('production','development')][string]$Engine = 'production',
    [ValidatePattern('^sm_[0-9]+$')][string]$Arch = 'sm_89',
    [ValidateSet('runtime','fold','short','ptx')][string]$GlBackend = 'ptx',
    [ValidateSet(0,4)][int]$OuterUnrollU = 0,
    [ValidateSet(0,1,2,3)][int]$AddSubMask = 0,
    [ValidateRange(1,64)][int]$SplitCompile = 1,
    [switch]$HostOnly,
    [switch]$Rebuild
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$vcvars = (Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue |
           Select-Object -First 1).FullName
if (-not $vcvars) { throw 'vcvars64.bat not found' }
if ($Engine -eq 'production' -and ($GlBackend -ne 'ptx' -or $OuterUnrollU -ne 0 -or $AddSubMask -ne 0)) {
    throw 'Production requires -GlBackend ptx -OuterUnrollU 0 -AddSubMask 0; use -Engine development for comparisons'
}
$cudaSource = if ($Engine -eq 'production') { 'src/cuda/ecm_cuda_stage2.cu' } else { 'tools/bench/ecm_cuda_stage2_dev.cu' }
$cudaStem = [IO.Path]::GetFileNameWithoutExtension($cudaSource)
$sources = @($cudaSource, 'src/core/ecm_cuda_stage2_main.cpp',
    'src/core/ecm_expr.cpp', 'src/core/ecm_worktodo.cpp', 'src/core/ecm_queue_config.cpp')
$deps = $sources + @('src/core/ecm_cuda_stage2.h', 'src/core/ecm_expr.h',
    'src/core/ecm_stage2_geometry.h', 'src/core/ecm_stage2_logging.h', 'src/core/ecm_stage2_fingerprint.h', 'src/cuda/ecm_stage2_tune.cuh',
    'src/core/ecm_stage2_factorize.h', 'src/core/ecm_stage2_cost_profile.h',
    'src/core/ecm_worktodo.h', 'src/core/ecm_queue_config.h', 'tools/build/build_ecm_cuda_stage2.ps1')
if ($Engine -eq 'production') {
    $deps += @(Get-ChildItem -LiteralPath 'src/cuda/stage2' -File -Filter '*.cuh' |
        Sort-Object Name | ForEach-Object { 'src/cuda/stage2/' + $_.Name })
} else {
    $deps += @(
    'tools/bench/stage2_tree_gpu.cu', 'tools/bench/stage2_d_model.cuh', 'tools/bench/ntt_poly_probe.cu', 'tools/bench/ntt_coop_outer.cuh', 'tools/bench/ntt_goldilocks_reduce.cuh','tools/bench/ntt_goldilocks_ptx.cuh','tools/bench/ntt_goldilocks_addsub.cuh',
    'tools/bench/stage2_baby_device.cuh', 'tools/bench/stage2_baby_host.cuh', 'tools/bench/stage2_point_mersenne.cuh', 'tools/bench/ntt_carry_partial.cuh',
    'tools/bench/stage2_giant_base_host.cuh','tools/bench/stage2_scaled_frontier.cuh','src/cuda/stage2/scale_plain.cuh')
}
$objDir = Join-Path $Build '_objects'
New-Item -ItemType Directory -Force $objDir | Out-Null
$exe = Join-Path $Build 'ecm_cuda_stage2.exe'
$signaturePath = Join-Path $objDir 'build_signature.txt'
$glMode = @{runtime=-1;fold=0;short=1;ptx=3}[$GlBackend]
$signature = @("arch=$Arch", "gl_backend=$GlBackend", "gl_fixed_mode=$glMode", "outer_unroll_u=$OuterUnrollU", (& nvcc --version | Out-String).Trim(), "split_compile=$SplitCompile", "engine=$Engine", "add_sub_mask=$AddSubMask")
$sourceHashes = [ordered]@{}
foreach ($dep in $deps) {
    $sourceHashes[$dep] = (Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash
    $signature += "$dep=$($sourceHashes[$dep])"
}
$signatureText = $signature -join "`n"
$cudaDeps = @($cudaSource,'src/core/ecm_cuda_stage2.h','src/core/ecm_stage2_geometry.h','src/core/ecm_stage2_logging.h',
    'src/cuda/ecm_stage2_tune.cuh') + @($deps | Where-Object { $_ -like 'tools/bench/*' -or $_ -like 'src/cuda/stage2/*' })
if ($HostOnly) {
    $previous = Get-Content -LiteralPath (Join-Path $Build 'build_manifest.json') -Raw | ConvertFrom-Json
    $previousSplit = if ($previous.split_compile) { $previous.split_compile } else { 1 }
    $previousAddSub = if ($null -ne $previous.add_sub_mask) { $previous.add_sub_mask } else { 0 }
    if ($previous.engine -ne $Engine -or $previous.architecture -ne $Arch -or $previous.gl_fixed_mode -ne $glMode -or
        $previous.outer_unroll_u -ne $OuterUnrollU -or $previousAddSub -ne $AddSubMask -or $previousSplit -ne $SplitCompile -or $previous.sources[4] -ne $signature[4]) {
        throw 'HostOnly requires identical CUDA architecture, backend, schedule and toolkit'
    }
    foreach ($dep in $cudaDeps) {
        if ($previous.source_hashes.$dep -ne $sourceHashes[$dep]) { throw "HostOnly CUDA dependency changed: $dep" }
    }
    $cudaObject = Join-Path $objDir "$cudaStem.obj"
    if (-not (Test-Path -LiteralPath $cudaObject)) { throw 'HostOnly CUDA object missing' }
    if ($previous.objects -and $previous.objects.$cudaStem -ne (Get-FileHash $cudaObject).Hash) {
        throw 'HostOnly CUDA object changed'
    }
    $cudaObjectHash = (Get-FileHash $cudaObject).Hash
}
$fresh = -not $Rebuild -and (Test-Path $exe) -and (Test-Path $signaturePath) -and
    ([IO.File]::ReadAllText((Resolve-Path $signaturePath)) -eq $signatureText)
if (-not $fresh) {
    $objects = @()
    foreach ($src in $sources) {
        $stem = [IO.Path]::GetFileNameWithoutExtension($src)
        $obj = Join-Path $objDir "$stem.obj"
        $log = Join-Path $objDir "$stem.log"
        $objects += $obj
        if ($HostOnly -and $src -eq $cudaSource) {
            Write-Host 'reuse CUDA object: matching compiled dependencies and toolkit'
            continue
        }
        $line = "call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch -DNTT_GL_FIXED_MODE=$glMode -DNTT_OUTER_UNROLL_U=$OuterUnrollU -DNTT_GL_ADD_SUB_MASK=$AddSubMask " +
            "--split-compile=$SplitCompile " +
            "-I third_party/gmp-zen3/dist/include -Xcompiler /utf-8 -Xcompiler /wd4819 " +
            "-c `"$src`" -o `"$obj`" > `"$log`" 2>&1"
        $sw = [Diagnostics.Stopwatch]::StartNew()
        & cmd.exe /c $line
        $code = $LASTEXITCODE
        $sw.Stop()
        Write-Host ("compile {0}: {1:N1}s exit={2}" -f $stem, $sw.Elapsed.TotalSeconds, $code)
        if ($code -ne 0) { Get-Content $log -Tail 40; throw "compile failed: $src" }
    }
    $objArgs = ($objects | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $linkLog = Join-Path $objDir 'link.log'
    $line = "call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch " +
        "$objArgs -L third_party/gmp-zen3/dist/lib -lgmp -o `"$exe`" > `"$linkLog`" 2>&1"
    & cmd.exe /c $line
    if ($LASTEXITCODE -ne 0) { Get-Content $linkLog -Tail 40; throw 'link failed' }
    foreach ($dep in $deps) {
        if ((Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash -ne $sourceHashes[$dep]) {
            throw "Source changed during build: $dep"
        }
    }
    if ($HostOnly -and (Get-FileHash $cudaObject).Hash -ne $cudaObjectHash) { throw 'Reused CUDA object changed during host build' }
    [IO.File]::WriteAllText((Join-Path (Resolve-Path $objDir) 'build_signature.txt'), $signatureText)
} else { Write-Host 'Source/toolkit/architecture signature unchanged; executable reused.' }
Copy-Item -LiteralPath 'third_party/gmp-zen3/dist/bin/gmp-10.dll' -Destination $Build -Force
$objectHashes = [ordered]@{}
foreach ($src in $sources) {
    $stem = [IO.Path]::GetFileNameWithoutExtension($src)
    $objectHashes[$stem] = (Get-FileHash -LiteralPath (Join-Path $objDir "$stem.obj")).Hash
}
$manifest = [ordered]@{
    exe = (Resolve-Path $exe).Path
    sha256 = (Get-FileHash $exe -Algorithm SHA256).Hash
    architecture = $Arch
    engine = $Engine
    gl_backend = $GlBackend
    gl_fixed_mode = $glMode
    outer_unroll_u = $OuterUnrollU
    add_sub_mask = $AddSubMask
    split_compile = $SplitCompile
    sources = $signature
    source_hashes = $sourceHashes
    objects = $objectHashes
    host_only = [bool]$HostOnly
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $Build 'build_manifest.json')
Write-Host ("built {0} ({1:N1} MB)" -f $exe, ((Get-Item $exe).Length / 1MB))
exit 0
