#Requires -Version 5.1
<#
.SYNOPSIS
Build the standalone CUDA Stage2 save/queue executable (MSVC + nvcc + GMP).
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_ecm_cuda_stage2.ps1 -Arch sm_89
#>
param(
    [string]$Build = 'build_cuda_cmake/production_stage2',
    [ValidatePattern('^sm_[0-9]+$')][string]$Arch = 'sm_89',
    [ValidateSet('runtime','fold','short','ptx')][string]$GlBackend = 'runtime',
    [switch]$Rebuild
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$vcvars = (Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat' -ErrorAction SilentlyContinue |
           Select-Object -First 1).FullName
if (-not $vcvars) { throw 'vcvars64.bat not found' }
$sources = @('src/cuda/ecm_cuda_stage2.cu', 'src/core/ecm_cuda_stage2_main.cpp',
    'src/core/ecm_expr.cpp', 'src/core/ecm_worktodo.cpp', 'src/core/ecm_queue_config.cpp')
$deps = $sources + @('src/core/ecm_cuda_stage2.h', 'src/core/ecm_expr.h',
    'src/core/ecm_worktodo.h', 'src/core/ecm_queue_config.h',
    'tools/bench/stage2_tree_gpu.cu', 'tools/bench/stage2_d_model.cuh', 'tools/bench/ntt_poly_probe.cu', 'tools/bench/ntt_coop_outer.cuh', 'tools/bench/ntt_goldilocks_reduce.cuh','tools/bench/ntt_goldilocks_ptx.cuh',
    'tools/bench/stage2_baby_device.cuh', 'tools/bench/stage2_baby_host.cuh',
    'tools/build/build_ecm_cuda_stage2.ps1')
$objDir = Join-Path $Build '_objects'
New-Item -ItemType Directory -Force $objDir | Out-Null
$exe = Join-Path $Build 'ecm_cuda_stage2.exe'
$signaturePath = Join-Path $objDir 'build_signature.txt'
$glMode = @{runtime=-1;fold=0;short=1;ptx=3}[$GlBackend]
$signature = @("arch=$Arch", "gl_backend=$GlBackend", "gl_fixed_mode=$glMode", (& nvcc --version | Out-String).Trim())
foreach ($dep in $deps) { $signature += "$dep=$((Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash)" }
$signatureText = $signature -join "`n"
$fresh = -not $Rebuild -and (Test-Path $exe) -and (Test-Path $signaturePath) -and
    ([IO.File]::ReadAllText((Resolve-Path $signaturePath)) -eq $signatureText)
if (-not $fresh) {
    $objects = @()
    foreach ($src in $sources) {
        $stem = [IO.Path]::GetFileNameWithoutExtension($src)
        $obj = Join-Path $objDir "$stem.obj"
        $log = Join-Path $objDir "$stem.log"
        $objects += $obj
        $line = "call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch -DNTT_GL_FIXED_MODE=$glMode " +
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
    [IO.File]::WriteAllText((Join-Path (Resolve-Path $objDir) 'build_signature.txt'), $signatureText)
} else { Write-Host 'Source/toolkit/architecture signature unchanged; executable reused.' }
Copy-Item -LiteralPath 'third_party/gmp-zen3/dist/bin/gmp-10.dll' -Destination $Build -Force
$manifest = [ordered]@{
    exe = (Resolve-Path $exe).Path
    sha256 = (Get-FileHash $exe -Algorithm SHA256).Hash
    architecture = $Arch
    gl_backend = $GlBackend
    gl_fixed_mode = $glMode
    sources = $signature
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $Build 'build_manifest.json')
Write-Host ("built {0} ({1:N1} MB)" -f $exe, ((Get-Item $exe).Length / 1MB))
exit 0
