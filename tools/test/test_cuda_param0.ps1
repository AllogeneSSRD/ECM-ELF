# ---------------------------------------------------------------------------
# test_cuda_param0.ps1 -- Suyama param0 on the CUDA/CGBN path (gpu_param = 0).
#
# Checks, in this order:
#   A. correctness  : GPU param0 and CPU param0 (--method mont --backend gmp) run
#                     the SAME curves for the same sigma -> save content identical
#                     (M991, B1=1e5, 64 curves)
#   B. 64-bit sigma : the same with sigma > 2^32 -- the batch path (gpu_param = 3)
#                     cannot do this at all, param0 must not truncate it
#   C. interop      : the GPU param0 save is accepted by gmp-ecm -resume (it checks
#                     each line's checksum against N, so a wrong N field or a wrong
#                     parametrization is caught immediately)
#   D. resume       : hard-kill a param0 run and rerun the same command line -> the
#                     save content is identical to an uninterrupted run
#                     (M3217, 4096 curves, B1=1e5)
#   E. no regression: param3 is unchanged -- CUDA and OpenCL agree line for line
#
# usage:
#   powershell -NoProfile -File tools\test\test_cuda_param0.ps1 `
#       [-CudaExe <path>] [-CpuExe <path>] [-GmpEcmExe <path>] [-SkipSlow]
#
# Defaults point at this repository's usual build trees:
#   build_cuda_cmake\ecm_cuda.exe   (full CGBN kernel set, see README)
#   build_vs18\Release\ecm.exe      (OpenCL backend)
# ---------------------------------------------------------------------------
param(
    [string]$CudaExe = 'build_cuda_cmake\ecm_cuda.exe',
    [string]$CpuExe  = 'build_vs18\Release\ecm.exe',
    [string]$GmpEcmExe = 'D:\code\GIMPS\gmp-ecm\ecm-2025.10.28-win.multiarch\ecm-zen3.exe',
    [switch]$SkipSlow
)

$ErrorActionPreference = 'Continue'   # native tools write to stderr; Stop would abort on that
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
function Resolve-Exe([string]$p) {
    if ([System.IO.Path]::IsPathRooted($p)) { return $p }
    return (Join-Path $root $p)
}
$CudaExe = Resolve-Exe $CudaExe
$CpuExe  = Resolve-Exe $CpuExe
foreach ($e in @($CudaExe, $CpuExe)) {
    if (-not (Test-Path $e)) { throw "missing executable: $e (build it first, see README)" }
}

$work = Join-Path $root 'build_vs18\test_cuda_param0'
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Path $work | Out-Null

$fails = 0
function Check([bool]$ok, [string]$what) {
    if ($ok) { Write-Host "  [PASS] $what" -ForegroundColor Green }
    else     { Write-Host "  [FAIL] $what" -ForegroundColor Red; $script:fails++ }
}

# save lines with the volatile WHO=/TIME= fields removed, so two runs are comparable
function Norm-Save([string]$f) {
    if ([string]::IsNullOrEmpty($f) -or -not (Test-Path $f)) { return @() }
    return Get-Content $f | Where-Object { $_ -match 'SIGMA=' } | ForEach-Object {
        (($_ -replace ' WHO=[^;]*;', '') -replace ' TIME=[^;]*;', '').Trim()
    }
}
function Compare-Saves([string]$a, [string]$b) {
    $na = Norm-Save $a; $nb = Norm-Save $b
    $n = [Math]::Min($na.Count, $nb.Count)
    $same = 0
    for ($i = 0; $i -lt $n; $i++) { if ($na[$i] -eq $nb[$i]) { $same++ } }
    return @{ a = $na.Count; b = $nb.Count; same = $same; diff = ($n - $same) }
}
function Write-N($file, $expr) {
    [System.IO.File]::WriteAllText($file, "$expr`n", ([System.Text.Encoding]::ASCII))
}
# run with N fed through a file (avoids the BOM a PowerShell pipe would prepend)
function Run-Tool([string]$exe, [string]$toolArgs, [string]$nfile, [string]$cwd) {
    # NOTE: never name a parameter $args -- it is a PowerShell automatic variable
    # and the parameter is silently ignored (this bit the script once already).
    # Also: the tools write progress to stderr and Windows PowerShell turns native
    # stderr into ErrorRecords, which under ErrorActionPreference=Stop would abort
    # the whole test -- hence the local Continue.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        # The working directory matters: relative outputs (-savea out.save, checkpoint
        # files) land in it, so a run that is meant to write into $cwd must actually
        # run there.
        if ([string]::IsNullOrEmpty($cwd)) {
            $out = cmd /c "`"$exe`" $toolArgs < `"$nfile`"" 2>&1
        } else {
            $out = cmd /c "cd /d `"$cwd`" && `"$exe`" $toolArgs < `"$nfile`"" 2>&1
        }
    } finally {
        $ErrorActionPreference = $prev
    }
    return ($out | Out-String)
}

$n991  = Join-Path $work 'n991.txt'
$n3217 = Join-Path $work 'n3217.txt'
Write-N $n991  '(2^991-1)'
Write-N $n3217 '(2^3217-1)'

Write-Host "=== A. GPU param0 vs CPU param0 (M991, B1=1e5, 64 curves) ==="
$gA = Join-Path $work 'A_gpu.save'; $cA = Join-Path $work 'A_cpu'
New-Item -ItemType Directory -Force $cA | Out-Null
$o = Run-Tool $CudaExe "-gpu -d 0 --gpu-param 0 -sigma 100000 -gpucurves 64 -savea $gA 1e5 0" $n991 $work
Check ($o -match 'parametrization = Suyama param0') 'GPU reports the Suyama param0 parametrization'
$null = Run-Tool $CpuExe "--method mont --backend gmp -sigma 100000 -gpucurves 64 --tmp-dir `"$cA`" --ckpt 0 1e5 0" $n991 $work
$cAfile = (Get-ChildItem $cA -Filter *.save | Select-Object -First 1).FullName
$r = Compare-Saves $gA $cAfile
Check (($r.a -eq 64) -and ($r.b -eq 64)) "both saves have 64 curve lines (GPU $($r.a), CPU $($r.b))"
Check ($r.same -eq 64 -and $r.diff -eq 0) "GPU and CPU agree on all 64 curves (sigma AND x)"

Write-Host ""
Write-Host "=== B. 64-bit sigma (> 2^32) ==="
$SIG = '9007199254740881'
$gB = Join-Path $work 'B_gpu.save'; $cB = Join-Path $work 'B_cpu'
New-Item -ItemType Directory -Force $cB | Out-Null
$o = Run-Tool $CudaExe "-v -gpu -d 0 --gpu-param 0 -sigma $SIG -gpucurves 32 -savea $gB 1e4 0" $n991 $work
Check ($o -match "sigma=$SIG") "GPU reports the full 64-bit sigma ($SIG), not a truncated one"
$null = Run-Tool $CpuExe "--method mont --backend gmp -sigma $SIG -gpucurves 32 --tmp-dir `"$cB`" --ckpt 0 1e4 0" $n991 $work
$cBfile = (Get-ChildItem $cB -Filter *.save | Select-Object -First 1).FullName
$r = Compare-Saves $gB $cBfile
Check ($r.same -eq 32 -and $r.diff -eq 0) "GPU and CPU agree on all 32 curves at sigma > 2^32"
$firstSigma = ((Get-Content $gB | Where-Object { $_ -match 'SIGMA=' } | Select-Object -First 1) -split '; ' |
               Where-Object { $_ -like 'SIGMA=*' })
Check ($firstSigma -eq "SIGMA=$SIG") "the save carries the untruncated sigma"

Write-Host ""
Write-Host "=== C. gmp-ecm accepts the param0 save (resume + stage 2) ==="
if (Test-Path $GmpEcmExe) {
    $o = & $GmpEcmExe -resume $gA 1e5 1e6 2>&1 | Out-String
    Check (-not ($o -match 'bad checksum')) 'no "bad checksum" complaint (the N field is the real N)'
    Check ($o -match 'Input number is \(2\^991-1\)') 'gmp-ecm read N = (2^991-1)'
    Check ($o -match 'sigma=0:100000') 'gmp-ecm read the line as param 0 (Suyama) with our sigma'
} else {
    Write-Host "  [SKIP] gmp-ecm not found at $GmpEcmExe" -ForegroundColor Yellow
}

if (-not $SkipSlow) {
    Write-Host ""
    Write-Host "=== D. hard kill + resume (M3217, 4096 curves, B1=1e5, sigma > 2^32) ==="
    $refDir = Join-Path $work 'D_ref'; $resDir = Join-Path $work 'D_res'
    New-Item -ItemType Directory -Force $refDir, $resDir | Out-Null
    $crit = 'stage1 returned'
    $o = Run-Tool $CudaExe "-gpu -d 0 --gpu-param 0 -sigma 9007199254740847 -gpucurves 4096 -savea out.save --ckpt 0 1e5 0" $n3217 $refDir
    Check ($o -match $crit) 'reference run completed'

    # A foreground run cannot be interrupted from inside the script, so the killed run
    # is a child process that we terminate after 25 s (the run needs ~64 s).
    # N goes in through a file redirect handled by cmd: .NET's RedirectStandardInput
    # writer prepends a UTF-8 BOM to the first write, which the driver's expression
    # parser rejects, and the run would exit before any checkpoint is written.
    Remove-Item (Join-Path $resDir 'out.save') -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $resDir '*.dat') -ErrorAction SilentlyContinue
    $nin = Join-Path $resDir 'nin.txt'
    Write-N $nin '(2^3217-1)'
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $env:ComSpec
    $psi.WorkingDirectory = $resDir
    $psi.UseShellExecute = $false
    $psi.Arguments = "/c `"`"$CudaExe`" -gpu -d 0 --gpu-param 0 -sigma 9007199254740847 -gpucurves 4096 " +
                     "-savea out.save --ckpt 5 1e5 0 < `"$nin`" > run.log 2>&1`""
    $p = [System.Diagnostics.Process]::Start($psi)
    Start-Sleep -Seconds 25
    $alive = -not $p.HasExited
    if ($alive) {
        # cmd.exe's child (ecm_cuda.exe) survives a plain $p.Kill(), and
        # "taskkill /T" did not take it down either, so the compute process is
        # stopped by name.  This test assumes no other ecm_cuda run is in progress.
        Get-Process -Name 'ecm_cuda' -ErrorAction SilentlyContinue | Stop-Process -Force
        if (-not $p.HasExited) { $p.Kill() }
    }
    $p.WaitForExit()
    Start-Sleep -Milliseconds 500
    Check $alive 'the run was still going after 25 s (so the kill is a real mid-run kill)'
    $ck = @(Get-ChildItem $resDir -Filter '*.dat' -ErrorAction SilentlyContinue)
    Check ($ck.Count -ge 1) 'the killed run left a checkpoint behind'
    if ($ck.Count -ge 1) {
        Check ($ck[0].Length -gt 1000000) "the checkpoint holds the whole 7-word/curve buffer ($($ck[0].Length) bytes)"
    } else {
        Check $false 'checkpoint size check (no checkpoint to inspect)'
    }

    $o = Run-Tool $CudaExe "-gpu -d 0 --gpu-param 0 -sigma 9007199254740847 -gpucurves 4096 -savea out.save --ckpt 5 1e5 0" $n3217 $resDir
    Check ($o -match $crit) 'the resumed run completed'
    $r = Compare-Saves (Join-Path $refDir 'out.save') (Join-Path $resDir 'out.save')
    Check ($r.a -eq 4096 -and $r.same -eq 4096 -and $r.diff -eq 0) "killed+resumed save is identical to the uninterrupted one ($($r.same)/$($r.a) lines)"
    # v4 checkpoint header: a 64-bit sigma must survive the round trip (that is the whole
    # point of the format bump -- v3 could only store the low 32 bits).
    $firstLine = (Get-Content (Join-Path $resDir 'out.save') | Where-Object { $_ -match 'SIGMA=' } |
                  Select-Object -First 1)
    Check ($firstLine -match 'SIGMA=9007199254740847') 'the resumed save still carries the full 64-bit sigma'
    $left = @(Get-ChildItem $resDir -Filter '*.dat' -ErrorAction SilentlyContinue)
    Check ($left.Count -eq 0) 'the checkpoint is removed once the save is written'
}

Write-Host ""
Write-Host "=== E. param3 regression: CUDA vs OpenCL ==="
$gE = Join-Path $work 'E_cuda.save'; $oE = Join-Path $work 'E_ocl.save'
$o1 = Run-Tool $CudaExe "-gpu -d 0 --gpu-param 3 -sigma 3:8888 -gpucurves 32 -savea $gE 1e5 0" $n991 $work
$o2 = Run-Tool $CpuExe  "-gpu -d 0 --gpu-param 3 -sigma 3:8888 -gpucurves 32 -savea $oE 1e5 0" $n991 $work
$r = Compare-Saves $gE $oE
Check ($r.same -eq 32 -and $r.diff -eq 0) "param3 results unchanged across the two backends (32/32 lines)"
$o3 = Run-Tool $CpuExe "-gpu -d 0 --gpu-param 0 -gpucurves 8 1e3 0" $n991 $work
Check ($o3 -match 'not implemented for the OpenCL backend') 'the OpenCL backend refuses gpu_param = 0 loudly'

Write-Host ""
if ($fails -eq 0) { Write-Host 'ALL OK' -ForegroundColor Green } else { Write-Host "FAILURES: $fails" -ForegroundColor Red }
exit $fails
