<#
.SYNOPSIS
    Compile the CUDA translation units of a CMake build directory in parallel.

.DESCRIPTION
    This repository's build directories use the "NMake Makefiles" generator, which
    compiles serially (a full ecm_cuda build takes roughly 20 minutes).  The three
    obvious parallel routes all fail on this machine: Ninja hangs, the Visual Studio
    / MSBuild generator cancels the CUDA build (MSB5021), and jom cannot configure
    ("parallel job execution disabled for Makefile").  They share one root cause:
    they all wrap nvcc in cmd child processes, and this environment kills such
    wrapped children.  Direct nvcc invocations work fine.

    This script therefore does the parallel part itself:

      1. configure (only with -Reconfigure, or when compile_commands.json is missing)
      2. read the exact nvcc command lines out of compile_commands.json
      3. launch them concurrently (each in its own cmd + vcvars64 process)
      4. run cmake --build once so that nmake only compiles host code and links

    Because step 3 writes the very object files nmake is about to want, step 4 is
    normally just a link.  Step 3 always recompiles every selected .cu file: use
    -Only to restrict it, or -SkipUpToDate to skip translation units whose object
    is newer than the sources and headers it was built from.

.PARAMETER BuildDir
    Build directory, relative to the repository root unless rooted.

.PARAMETER Target
    CMake target passed to the final cmake --build.

.PARAMETER Jobs
    Maximum concurrent nvcc processes.  Default: min(6, logical CPUs).

.PARAMETER Only
    Regex the .cu source path must match (default: every .cu).

.PARAMETER Reconfigure
    Re-run cmake configure first (adds -DCMAKE_EXPORT_COMPILE_COMMANDS=ON).

.PARAMETER SkipUpToDate
    Skip a translation unit whose .obj is newer than its source, the kernel headers
    and CMakeLists.txt.  Faster for host-only edits, but it trusts timestamps: after
    changing a CMake option, pass -Reconfigure or drop this switch.

.PARAMETER NoBuild
    Stop after the parallel compiles (do not run the final cmake --build).

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\parallel_nvcc.ps1 `
        -BuildDir build_cuda_cmake -Jobs 6

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\parallel_nvcc.ps1 `
        -BuildDir build_nm16 -Only tpi16 -Reconfigure

.NOTES
    Logs (one .log per translation unit) are written to <BuildDir>\par_nvcc\.
    See docs/ECM_CGBN_OPTIMIZATION.md section 8 item 8.
#>
param(
    [string]$BuildDir = "build_cuda_cmake",
    [string]$Target = "ecm_cuda",
    [int]$Jobs = 0,
    [string]$Only = "",
    [switch]$Reconfigure,
    [switch]$SkipUpToDate,
    [switch]$NoBuild,
    [string]$VcVars = "",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not [System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
if (-not (Test-Path $BuildDir)) { throw "build dir not found: $BuildDir" }

if (-not $VcVars) {
    $candidates = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending
    if (-not $candidates) { throw "vcvars64.bat not found; pass -VcVars" }
    $VcVars = $candidates[0].FullName
}
if (-not (Test-Path $VcVars)) { throw "vcvars64.bat not found: $VcVars" }

$cmakeFile = Join-Path $BuildDir "compile_commands.json"
if ($Reconfigure -or -not (Test-Path $cmakeFile)) {
    Write-Host "configuring $BuildDir (exporting compile_commands.json) ..."
    $cfg = "call `"$VcVars`" >nul 2>&1 && cmake -S `"$repo`" -B `"$BuildDir`" -DCMAKE_BUILD_TYPE=$Config -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    & cmd.exe /c $cfg
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }
    if (-not (Test-Path $cmakeFile)) { throw "cmake did not write compile_commands.json" }
}

$entries = (Get-Content $cmakeFile -Raw | ConvertFrom-Json) |
    Where-Object { $_.file -match '\.cu$' } |
    Where-Object { -not $Only -or $_.file -match $Only }
if (-not $entries) { throw "no .cu entries matched (-Only '$Only')" }

if ($Jobs -le 0) { $Jobs = [Math]::Min(6, [Environment]::ProcessorCount) }
$Jobs = [Math]::Max(1, [Math]::Min($Jobs, @($entries).Count))

$logDir = Join-Path $BuildDir "par_nvcc"
New-Item -ItemType Directory -Force $logDir | Out-Null

# --- decide what to compile -------------------------------------------------
$newestSource = Get-Date "1970-01-01"
foreach ($pat in @("kernels\cuda\*", "include\*", "CMakeLists.txt", "kernels\*.h")) {
    $p = Join-Path $repo $pat
    Get-Item $p -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.LastWriteTime -gt $newestSource) { $newestSource = $_.LastWriteTime }
    }
}

function Get-ObjPath([string]$command, [string]$dir) {
    $m = [regex]::Match($command, '-o\s+("[^"]+"|\S+)')
    if (-not $m.Success) { return $null }
    $o = $m.Groups[1].Value.Trim('"')
    if (-not [System.IO.Path]::IsPathRooted($o)) { $o = Join-Path $dir $o }
    return $o
}

$tuList = @()
foreach ($e in $entries) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($e.file)
    $obj = Get-ObjPath $e.command $e.directory
    $skip = $false
    if ($SkipUpToDate -and $obj -and (Test-Path $obj)) {
        $objTime = (Get-Item $obj).LastWriteTime
        $srcTime = (Get-Item $e.file).LastWriteTime
        if ($objTime -gt $srcTime -and $objTime -gt $newestSource) { $skip = $true }
    }
    $tuList += [pscustomobject]@{
        Name = $name; Entry = $e; Obj = $obj; Skip = $skip
        Bat = Join-Path $logDir "$name.bat"
        Out = Join-Path $logDir "$name.log"
        Err = Join-Path $logDir "$name.err.log"
        Proc = $null; Start = $null; Seconds = 0.0
    }
}

$todo = @($tuList | Where-Object { -not $_.Skip })
Write-Host ("parallel nvcc: {0} of {1} translation units, {2} concurrent (logs in {3})" -f `
    $todo.Count, $tuList.Count, $Jobs, $logDir)
if ($todo.Count -eq 0) { Write-Host "nothing to do" }

# --- run them, at most $Jobs at a time --------------------------------------
$queue = New-Object System.Collections.Queue
foreach ($j in $todo) { $queue.Enqueue($j) }
$running = New-Object System.Collections.ArrayList
$failed = @()

while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($queue.Count -gt 0 -and $running.Count -lt $Jobs) {
        $j = $queue.Dequeue()
        $bat = "@echo off`r`ncall `"$VcVars`" >nul 2>&1`r`ncd /d `"$($j.Entry.directory)`"`r`n$($j.Entry.command)`r`n"
        [System.IO.File]::WriteAllText($j.Bat, $bat, [System.Text.Encoding]::ASCII)
        $j.Start = Get-Date
        $j.Proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$($j.Bat)`"" -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput $j.Out -RedirectStandardError $j.Err
        [void]$running.Add($j)
    }
    Start-Sleep -Milliseconds 200
    foreach ($j in @($running)) {
        if ($j.Proc.HasExited) {
            $j.Seconds = ((Get-Date) - $j.Start).TotalSeconds
            [void]$running.Remove($j)
            # Start-Process -PassThru frequently leaves ExitCode $null (and "$null -ne 0" is TRUE
            # in PowerShell, which made every SUCCESSFUL compile report as a failure).  Read the
            # real code when it is available and otherwise fall back to "did nvcc write a fresh
            # object file?", which is the property we actually care about.
            $code = $null
            try { $j.Proc.WaitForExit(); $code = $j.Proc.ExitCode } catch { $code = $null }
            $objFresh = ($j.Obj -and (Test-Path $j.Obj) -and ((Get-Item $j.Obj).LastWriteTime -ge $j.Start))
            if ($null -eq $code) { $code = if ($objFresh) { 0 } else { 1 } }
            $objMissing = ($j.Obj -and -not $objFresh)
            if ($code -ne 0 -or $objMissing) {
                $failed += $j
                $tail = (Get-Content $j.Out -Tail 12 -ErrorAction SilentlyContinue) -join "`n"
                Write-Host ("  FAIL {0} ({1:N1}s, exit {2})`n{3}" -f $j.Name, $j.Seconds, $code, $tail)
            } else {
                Write-Host ("  ok   {0} ({1:N1}s)" -f $j.Name, $j.Seconds)
            }
            if ($objMissing) { Write-Host ("       (no object file at {0})" -f $j.Obj) }
        }
    }
}

foreach ($j in $tuList | Where-Object { $_.Skip }) { Write-Host ("  skip {0} (up to date)" -f $j.Name) }

$wall = 0.0
$slowest = $null
foreach ($j in $tuList) { if ($j.Seconds -gt $wall) { $wall = $j.Seconds; $slowest = $j.Name } }
$sum = ($tuList | Measure-Object -Property Seconds -Sum).Sum
Write-Host ("compiled {0} TU(s): wall {1:N1}s (serial sum {2:N1}s, speedup {3:N2}x)" -f `
    $todo.Count, $wall, $sum, $(if ($wall -gt 0) { $sum / $wall } else { 0 }))
if ($slowest) {
    Write-Host ("critical path = {0} ({1:N1}s): further speedup means splitting THAT TU" -f $slowest, $wall)
}

if ($failed.Count -gt 0) {
    Write-Host ("{0} translation unit(s) FAILED; see {1}" -f $failed.Count, $logDir)
    exit 1
}

if ($NoBuild) { exit 0 }

# --- hand the rest (host objects + link) to nmake ---------------------------
Write-Host "cmake --build $BuildDir --target $Target (host code + link) ..."
$build = "call `"$VcVars`" >nul 2>&1 && cmake --build `"$BuildDir`" --config $Config --target $Target"
& cmd.exe /c $build
if ($LASTEXITCODE -ne 0) { throw "cmake --build failed ($LASTEXITCODE)" }
Write-Host "done"
