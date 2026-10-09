#Requires -Version 5.1
<#
.SYNOPSIS
Build the standalone CUDA Stage2 save/queue executable (MSVC + nvcc + GMP).
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_stage2_local.ps1 -Arch sm_89
#>
param(
    [string]$Build = 'build_cuda_cmake/production_stage2',
    [ValidateSet('production','development')][string]$Engine = 'production',
    [ValidatePattern('^sm_[0-9]+$')][string]$Arch = 'sm_89',
    [ValidateSet('runtime','fold','short','ptx')][string]$GlBackend = 'ptx',
    [ValidateSet(0,4)][int]$OuterUnrollU = 0,
    [ValidateSet(0,1,2,3)][int]$AddSubMask = 0,
    [ValidateRange(1,64)][int]$SplitCompile = 8,
    [string]$Gmp = '',
    [string]$CudaRoot = '',
    [string]$VcVars = '',
    [switch]$HostOnly,
    [switch]$Rebuild
)
$ErrorActionPreference = 'Stop'
if (-not $PSBoundParameters.ContainsKey('AddSubMask') -and $Engine -eq 'production') { $AddSubMask = 1 }
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$buildTimer = [Diagnostics.Stopwatch]::StartNew()
. (Join-Path $PSScriptRoot 'internal/gmp_dependency.ps1')
$gmpDependency = Resolve-EcmGmp -Repo $repo -Gmp $Gmp
$Gmp = $gmpDependency.Root
. (Join-Path $PSScriptRoot 'internal/cuda_toolchain.ps1')
$cuda = Resolve-EcmCudaToolkit -Archs $Arch -CudaRoot $CudaRoot
$vc = Resolve-EcmVcEnvironment $cuda -VcVars $VcVars
$Build = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Build)
Write-Host '== Stage2 build =='
Write-Host "   engine    : $Engine"
Write-Host "   GPU       : $Arch"
Write-Host "   NTT       : backend=$GlBackend add/sub=$AddSubMask outer_unroll=$OuterUnrollU"
Write-Host "   parallel  : nvcc split compile=$SplitCompile"
Write-Host "   build dir : $Build"
Write-Host "   GMP       : $Gmp"
Write-Host "   CUDA      : $($cuda.Root)"
Write-Host "   MSVC      : $($vc.Version)"
& cmake "-DECM_CONFIG_ROOT=$repo" -P (Join-Path $PSScriptRoot 'internal/check_ecm_config.cmake')
if ($LASTEXITCODE -ne 0) { throw 'Generated ECM config is stale; run python tools/gen/generate_ecm_config.py' }
if ($Engine -eq 'production' -and ($GlBackend -ne 'ptx' -or $OuterUnrollU -ne 0 -or $AddSubMask -ne 1)) {
    throw 'Production requires -GlBackend ptx -OuterUnrollU 0 -AddSubMask 1; use -Engine development for comparisons'
}
$cudaSource = if ($Engine -eq 'production') { 'src/cuda/ecm_cuda_stage2.cu' } else { 'tools/bench/ecm_cuda_stage2_dev.cu' }
$cudaStem = [IO.Path]::GetFileNameWithoutExtension($cudaSource)
$sources = @($cudaSource, 'src/core/ecm_cuda_stage2_main.cpp',
    'src/core/ecm_expr.cpp', 'src/core/ecm_worktodo.cpp', 'src/core/ecm_queue_config.cpp')
$deps = $sources + @('src/core/ecm_cuda_stage2.h', 'src/core/ecm_expr.h',
    'src/core/ecm_stage2_geometry.h', 'src/core/ecm_stage2_requests.h', 'src/core/ecm_stage2_ntt_memory.h', 'src/core/ecm_stage2_giant_memory.h', 'src/core/ecm_stage2_s4_memory.h', 'src/core/ecm_stage2_s4_program.h', 'src/core/ecm_stage2_modulus.h', 'src/core/ecm_stage2_logging.h', 'src/core/ecm_stage2_console.h', 'src/core/ecm_stage2_queue_state.h', 'src/core/ecm_stage2_fingerprint.h', 'src/cuda/ecm_stage2_tune.cuh',
    'src/core/ecm_stage2_factorize.h', 'src/core/ecm_stage2_cost_profile.h', 'src/core/ecm_stage2_tune_format.h', 'src/core/ecm_stage2_workspace_memory.h',
    'src/core/ecm_worktodo.h', 'src/core/ecm_queue_config.h', 'src/core/ecm_ini.h',
    'src/core/generated/ecm_config_generated.h', 'src/core/generated/ecm_ini_template.h',
    'config/ecm_options.json', 'config/ecm_config.generated.json',
    'tools/gen/generate_ecm_config.py', 'tools/build/internal/check_ecm_config.cmake',
    'tools/build/build_stage2_local.ps1','tools/build/internal/gmp_dependency.ps1','tools/build/internal/cuda_toolchain.ps1')
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
$signature = @("arch=$Arch", "gl_backend=$GlBackend", "gl_fixed_mode=$glMode", "outer_unroll_u=$OuterUnrollU", $cuda.VersionText, "split_compile=$SplitCompile", "engine=$Engine", "add_sub_mask=$AddSubMask",
    "cuda_root=$($cuda.Root)","host_compiler=$($vc.Compiler)","host_version=$($vc.Version)")
$sourceHashes = [ordered]@{}
foreach ($dep in $deps) {
    $sourceHashes[$dep] = (Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash
    $signature += "$dep=$($sourceHashes[$dep])"
}
$gmpHashes = $gmpDependency.Hashes
$signature += "gmp=$Gmp"
foreach ($name in $gmpHashes.Keys) { $signature += "gmp/$name=$($gmpHashes[$name])" }
$signatureText = $signature -join "`n"
$cudaDeps = @($cudaSource,'src/core/ecm_cuda_stage2.h','src/core/ecm_stage2_geometry.h', 'src/core/ecm_stage2_requests.h','src/core/ecm_stage2_ntt_memory.h','src/core/ecm_stage2_giant_memory.h','src/core/ecm_stage2_s4_memory.h','src/core/ecm_stage2_s4_program.h','src/core/ecm_stage2_modulus.h','src/core/ecm_stage2_logging.h',
    'src/cuda/ecm_stage2_tune.cuh','src/core/ecm_stage2_workspace_memory.h') + @($deps | Where-Object { $_ -like 'tools/bench/*' -or $_ -like 'src/cuda/stage2/*' })
if ($HostOnly) {
    $previous = Get-Content -LiteralPath (Join-Path $Build 'build_manifest.json') -Raw | ConvertFrom-Json
    $previousSplit = if ($previous.split_compile) { $previous.split_compile } else { 1 }
    $previousAddSub = if ($null -ne $previous.add_sub_mask) { $previous.add_sub_mask } else { 0 }
    if ($previous.engine -ne $Engine -or $previous.architecture -ne $Arch -or $previous.gl_fixed_mode -ne $glMode -or
        $previous.outer_unroll_u -ne $OuterUnrollU -or $previousAddSub -ne $AddSubMask -or $previousSplit -ne $SplitCompile -or $previous.sources[4] -ne $signature[4] -or
        $previous.cuda_root -ne $cuda.Root -or $previous.host_compiler -ne $vc.Compiler -or $previous.host_version -ne $vc.Version) {
        throw 'HostOnly requires identical CUDA architecture, backend, schedule and toolkit'
    }
    if (-not $previous.gmp_hashes -or $previous.gmp_hashes.'include/gmp.h' -ne $gmpHashes['include/gmp.h']) {
        throw 'HostOnly requires the same recorded GMP header; rebuild old manifests or changed GMP headers normally'
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
    $compiled = 0
    $compileSeconds = 0.0
    Write-Host "== compile $($sources.Count) translation units =="
    foreach ($src in $sources) {
        $stem = [IO.Path]::GetFileNameWithoutExtension($src)
        $obj = Join-Path $objDir "$stem.obj"
        $log = Join-Path $objDir "$stem.log"
        $objects += $obj
        if ($HostOnly -and $src -eq $cudaSource) {
            Write-Host ("  skip {0} (verified CUDA object)" -f $stem)
            continue
        }
        $compiled++
        Write-Host ("  start [{0}/{1}] {2}" -f $compiled, ($sources.Count - [int][bool]$HostOnly), $src)
        $line = "$($vc.Setup) && `"$($cuda.Nvcc)`" -ccbin `"$($vc.Compiler)`" -std=c++17 -O3 -arch=$Arch -DNTT_GL_FIXED_MODE=$glMode -DNTT_OUTER_UNROLL_U=$OuterUnrollU -DNTT_GL_ADD_SUB_MASK=$AddSubMask " +
            "--split-compile=$SplitCompile " +
            "-I `"$Gmp/include`" -Xcompiler /utf-8 -Xcompiler /wd4819 " +
            "-c `"$src`" -o `"$obj`""
        $sw = [Diagnostics.Stopwatch]::StartNew()
        # Keep setup and compilation on separate lines. Capture via a pipe:
        # direct cmd file redirection crashes NVCC's compiler probe in some
        # restricted Windows execution environments, even on an empty source.
        $commandFile = Join-Path $objDir "$stem.compile.cmd"
        $compileCommand = $line.Substring(($vc.Setup + ' && ').Length)
        [IO.File]::WriteAllText($commandFile, "@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$compileCommand`r`n", [Text.UTF8Encoding]::new($false))
        & cmd.exe /d /c $commandFile 2>&1 | Out-File -LiteralPath $log -Encoding utf8
        $code = $LASTEXITCODE
        $sw.Stop()
        $compileSeconds += $sw.Elapsed.TotalSeconds
        if ($code -ne 0) {
            Write-Host ("  FAIL {0} ({1:N1}s, exit {2})" -f $stem, $sw.Elapsed.TotalSeconds, $code) -ForegroundColor Red
            Write-Host "       log: $log"
            Get-Content $log -Tail 40
            throw "compile failed: $src"
        }
        Write-Host ("  ok   {0,-26} ({1:N1}s)" -f $stem, $sw.Elapsed.TotalSeconds)
    }
    Write-Host ("compiled {0} TU(s): wall {1:N1}s (nvcc split compile={2})" -f $compiled, $compileSeconds, $SplitCompile)
    $objArgs = ($objects | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $linkLog = Join-Path $objDir 'link.log'
    Write-Host '== link ecm_cuda_stage2.exe =='
    $linkTimer = [Diagnostics.Stopwatch]::StartNew()
    $line = "$($vc.Setup) && `"$($cuda.Nvcc)`" -ccbin `"$($vc.Compiler)`" -std=c++17 -O3 -arch=$Arch " +
        "$objArgs -L `"$Gmp/lib`" -lgmp -o `"$exe`""
    $commandFile = Join-Path $objDir 'link.cmd'
    $linkCommand = $line.Substring(($vc.Setup + ' && ').Length)
    [IO.File]::WriteAllText($commandFile, "@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$linkCommand`r`n", [Text.UTF8Encoding]::new($false))
    & cmd.exe /d /c $commandFile 2>&1 | Out-File -LiteralPath $linkLog -Encoding utf8
    if ($LASTEXITCODE -ne 0) { Get-Content $linkLog -Tail 40; throw 'link failed' }
    $linkTimer.Stop()
    Write-Host ("  ok   link ({0:N1}s)" -f $linkTimer.Elapsed.TotalSeconds)
    foreach ($dep in $deps) {
        if ((Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash -ne $sourceHashes[$dep]) {
            throw "Source changed during build: $dep"
        }
    }
    if ($HostOnly -and (Get-FileHash $cudaObject).Hash -ne $cudaObjectHash) { throw 'Reused CUDA object changed during host build' }
    [IO.File]::WriteAllText((Join-Path (Resolve-Path $objDir) 'build_signature.txt'), $signatureText)
} else { Write-Host 'Source/toolkit/architecture signature unchanged; executable reused.' }
foreach ($name in $gmpHashes.Keys) {
    if ((Get-FileHash -LiteralPath (Join-Path $Gmp $name)).Hash -ne $gmpHashes[$name]) { throw "GMP changed during build: $name" }
}
Copy-Item -LiteralPath (Join-Path $Gmp 'bin/gmp-10.dll') -Destination $Build -Force
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
    cuda_root = $cuda.Root
    toolkit = $cuda.VersionText
    host_compiler = $vc.Compiler
    host_version = $vc.Version
    sources = $signature
    source_hashes = $sourceHashes
    objects = $objectHashes
    host_only = [bool]$HostOnly
    gmp_root = $Gmp
    gmp_hashes = $gmpHashes
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $Build 'build_manifest.json')
$buildTimer.Stop()
Write-Host '== produced =='
Write-Host ("   ecm_cuda_stage2.exe: {0:N0} B ({1:N1} MB)" -f (Get-Item $exe).Length, ((Get-Item $exe).Length / 1MB))
Write-Host "   executable : $exe"
Write-Host "   logs       : $objDir"
Write-Host ("   elapsed    : {0:N1}s" -f $buildTimer.Elapsed.TotalSeconds)
Write-Host 'done'
exit 0
