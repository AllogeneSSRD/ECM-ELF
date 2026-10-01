#Requires -Version 5.1
<#
.SYNOPSIS
    Build tools/bench/stage2_gpu_probe.cu (the CUDA/CGBN stage-2 probe) in parallel.

.DESCRIPTION
    The probe is NOT part of the CMake target, so tools/build/parallel_nvcc.ps1 (which
    works off the CMake build dir's compile_commands.json) cannot see it.  This is the
    same idea for the four stage-2 translation units: they have no dependency on each
    other, so they are compiled concurrently and only the link is serial.

      probe TU          tools/bench/stage2_gpu_probe.cu
      host TU           kernels/cuda/cgbn_stage2.cu
      kernel TUs        kernels/cuda/cgbn_stage2_kernels_tpi4.cu
                        kernels/cuda/cgbn_stage2_kernels_tpi8.cu

    Each TU gets its own log under <Build>/_s2probe/ so a failure can be read without
    rerunning the whole build, and up-to-date objects are skipped (a header change
    invalidates all of them, a .cu change only itself).  Use -Jobs 1 for a serial,
    easy-to-read run and -Force to ignore the up-to-date check.

.PARAMETER Jobs
    Concurrent nvcc invocations (default 4 = the number of TUs).  Each nvcc with
    -arch=sm_89 uses roughly 1.5 GB while it runs.

.PARAMETER Rebuild
    Recompile everything, even objects that are newer than their sources.

.EXAMPLE
    powershell -File tools\build\build_stage2_probe.ps1
    powershell -File tools\build\build_stage2_probe.ps1 -Jobs 2 -Rebuild
#>
param(
    [string]$Build = 'build_cuda_cmake',
    [string]$Arch = 'sm_89',
    [int]$Jobs = 4,
    [switch]$Rebuild
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo

$vcvars = (Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue |
           Select-Object -First 1).FullName
if (-not $vcvars) { throw "vcvars64.bat not found; install the MSVC build tools" }

# --- what to build ---------------------------------------------------------------
# Order matters only for the log: the link is always last and always serial.
$tus = @(
    @{ name = 'stage2_gpu_probe';          src = 'tools/bench/stage2_gpu_probe.cu' },
    @{ name = 'cgbn_stage2';               src = 'kernels/cuda/cgbn_stage2.cu' },
    @{ name = 'cgbn_stage2_kernels_tpi4';  src = 'kernels/cuda/cgbn_stage2_kernels_tpi4.cu' },
    @{ name = 'cgbn_stage2_kernels_tpi8';  src = 'kernels/cuda/cgbn_stage2_kernels_tpi8.cu' }
)

$inc = @(
    '-I kernels/cuda',
    '-I cgbn/include/cgbn',      # <cgbn.h> itself
    '-I cgbn/include',           # cgbn.h's own "arith/..." includes
    '-I include',                # ecm.h (via cuda_ecm_shim.h)
    '-I third_party/gmp-zen3/dist/include'
) -join ' '
$gmpLib = 'third_party/gmp-zen3/dist/lib'
$gmpDll = 'third_party/gmp-zen3/dist/bin/gmp-10.dll'

$objDir = Join-Path $Build '_s2probe'
$exe = Join-Path $Build 'stage2_gpu_probe.exe'
New-Item -ItemType Directory -Force $objDir | Out-Null

# A header change must invalidate every TU (they all include the kernel header).
$headers = @('kernels/cuda/cgbn_stage2_kernel.h', 'kernels/cuda/cgbn_stage2_cuda.h',
             'kernels/cuda/cuda_ecm_shim.h') |
    Where-Object { Test-Path $_ } | ForEach-Object { (Get-Item $_).LastWriteTime }
$newestHeader = if ($headers) { ($headers | Measure-Object -Maximum).Maximum } else { [datetime]'1970-01-01' }

$todo = @()
$skip = @()
foreach ($t in $tus) {
    $obj = Join-Path $objDir ($t.name + '.obj')
    $t.obj = $obj
    if (-not $Rebuild -and (Test-Path $obj)) {
        $objTime = (Get-Item $obj).LastWriteTime
        $srcTime = (Get-Item $t.src).LastWriteTime
        if ($objTime -gt $srcTime -and $objTime -gt $newestHeader) { $skip += $t; continue }
    }
    $todo += $t
}

Write-Host ("stage2 probe build: exe={0} arch={1} jobs={2}" -f $exe, $Arch, $Jobs)
Write-Host ("  {0} to compile, {1} up to date" -f $todo.Count, $skip.Count)
foreach ($t in $skip) { Write-Host ("  skip {0}" -f $t.name) }
if ($todo.Count -eq 0 -and (Test-Path $exe)) {
    Write-Host "  nothing to do; link output is up to date"
    exit 0
}

# --- compile ---------------------------------------------------------------------
# Each TU is compiled by a PowerShell job that shells out to cmd (which first loads
# vcvars64.bat).  Start-Process is deliberately NOT used: under the DSH file sandbox a
# child process created with -NoNewWindow is denied, while a plain `cmd /c` call from a
# job is not.  The job writes nvcc's output to its own log and returns exit code + time.
$compileBlock = {
    param($repo, $vcvars, $inc, $arch, $src, $obj, $log)
    # A PowerShell job starts in the user profile, NOT in the caller's directory, so the
    # relative paths below (sources, -I include dirs, object/log paths) only work after
    # this.  (Learned the hard way: without it the job dies with "cannot find the path".)
    Set-Location -LiteralPath $repo
    $line = "call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$arch $inc " +
            "-Xcompiler /wd4819 -c -o `"$obj`" `"$src`" > `"$log`" 2>&1"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & cmd.exe /c $line
    $code = $LASTEXITCODE
    $sw.Stop()
    [pscustomobject]@{ exit = $code; seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1) }
}

function Show-Tail([string]$name, [string]$log) {
    Write-Host ("--- {0} failed; tail of {1}" -f $name, $log) -ForegroundColor Red
    if (Test-Path $log) { Get-Content $log -Tail 25 | ForEach-Object { Write-Host ("    " + $_) } }
}

$results = @()
if ($Jobs -le 1) {
    foreach ($t in $todo) {
        $log = Join-Path $objDir ($t.name + '.log')
        $r = & $compileBlock $repo $vcvars $inc $Arch $t.src $t.obj $log
        Write-Host ("  {0,-28} {1,7:N1}s exit={2}" -f $t.name, $r.seconds, $r.exit)
        $results += [pscustomobject]@{ name = $t.name; exit = $r.exit; seconds = $r.seconds; log = $log }
    }
} else {
    $running = @()
    $queue = [System.Collections.Queue]::new()
    foreach ($t in $todo) { $queue.Enqueue($t) }
    $max = [Math]::Min($Jobs, $todo.Count)
    while ($queue.Count -gt 0 -or $running.Count -gt 0) {
        while ($queue.Count -gt 0 -and $running.Count -lt $max) {
            $t = $queue.Dequeue()
            $log = Join-Path $objDir ($t.name + '.log')
            $job = Start-Job -ScriptBlock $compileBlock -ArgumentList $repo, $vcvars, $inc, $Arch, $t.src, $t.obj, $log
            $running += [pscustomobject]@{ job = $job; name = $t.name; log = $log }
        }
        $done = @($running | Where-Object { $_.job.State -ne 'Running' })
        foreach ($d in $done) {
            $r = Receive-Job -Job $d.job
            Remove-Job -Job $d.job -Force
            $exitCode = if ($r) { $r.exit } else { 1 }
            $secs = if ($r) { $r.seconds } else { 0 }
            Write-Host ("  {0,-28} {1,7:N1}s exit={2}" -f $d.name, $secs, $exitCode)
            $results += [pscustomobject]@{ name = $d.name; exit = $exitCode; seconds = $secs; log = $d.log }
        }
        $running = @($running | Where-Object { $_.job.State -eq 'Running' })
        if ($running.Count -gt 0) { Start-Sleep -Milliseconds 500 }
    }
}

$failed = @($results | Where-Object { $_.exit -ne 0 })
foreach ($f in $failed) { Show-Tail $f.name $f.log }
if ($failed.Count -gt 0) { throw ("compile failed for: " + (($failed | ForEach-Object { $_.name }) -join ', ')) }

# --- link (serial) ---------------------------------------------------------------
$objs = ($tus | ForEach-Object { '"{0}"' -f $_.obj }) -join ' '
$linkLog = Join-Path $objDir 'link.log'
$linkLine = "call `"$vcvars`" >nul 2>&1 && nvcc -std=c++17 -O3 -arch=$Arch " +
            "-o `"$exe`" $objs -L $gmpLib -lgmp > `"$linkLog`" 2>&1"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& cmd.exe /c $linkLine
$linkCode = $LASTEXITCODE
$sw.Stop()
if ($linkCode -ne 0) {
    Show-Tail 'link' $linkLog
    throw "link failed (exit $linkCode)"
}
Write-Host ("  {0,-28} {1,7:N1}s exit=0" -f 'link', $sw.Elapsed.TotalSeconds)

# gmp-10.dll has to sit next to the exe (the same trap the other bench probes document).
if (Test-Path $gmpDll) { Copy-Item $gmpDll (Split-Path -Parent $exe) -Force }

$total = ($results | Measure-Object -Property seconds -Sum).Sum + $sw.Elapsed.TotalSeconds
$wall = ($results | Measure-Object -Property seconds -Maximum).Maximum
Write-Host ("built {0} ({1:N1} MB) -- cpu-sum {2:N1}s, slowest TU {3:N1}s" -f `
            $exe, ((Get-Item $exe).Length / 1MB), $total, $wall)
exit 0
