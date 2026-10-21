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
    normally just a link. Step 3 reuses matching objects by default. Use -Only
    to restrict the selection, or -Rebuild to compile every selected
    file. Command, dependency-content and object identities must all match.

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
    Reuse an object only when its compile command, response files, transitive
    dependency contents and recorded object hash all match. Old builds without
    dependency records compile once to establish their identity. Enabled by default.

.PARAMETER Rebuild
    Compile every selected translation unit even when its identity still matches.

.PARAMETER NoBuild
    Stop after the parallel compiles (do not run the final cmake --build).

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\internal\parallel_nvcc.ps1 `
        -BuildDir build_cuda_cmake -Jobs 6

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\internal\parallel_nvcc.ps1 `
        -BuildDir build_nm16 -Only tpi16 -Reconfigure

.NOTES
    Logs (one .log per translation unit) are written to <BuildDir>\par_nvcc\.
    See docs/performance/STAGE1.md item 8.
#>
param(
    [string]$BuildDir = "build_cuda_cmake",
    [string]$Target = "ecm_cuda",
    [int]$Jobs = 0,
    [string]$Only = "",
    [string]$SourceRegex = '\.cu$',
    [switch]$AllSources,
    [switch]$Reconfigure,
    [switch]$SkipUpToDate = $true,
    [switch]$Rebuild,
    [switch]$HostOnly,
    [switch]$NoBuild,
    [string]$VcVars = "",
    [ValidatePattern('^$|^[0-9.]+$')][string]$VcVersion = "",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
if ($HostOnly -and ($Rebuild -or -not $SkipUpToDate -or $Only -or $SourceRegex -ne '\.cu$')) {
    throw 'HostOnly requires verification of the full CUDA target and forbids forced compilation'
}
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
. (Join-Path $PSScriptRoot 'cuda_build_state.ps1')
& cmake "-DECM_CONFIG_ROOT=$repo" -P (Join-Path $PSScriptRoot 'check_ecm_config.cmake')
if ($LASTEXITCODE -ne 0) { throw 'Generated ECM config is stale; run python tools/gen/generate_ecm_config.py' }
if (-not [System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
if (-not (Test-Path $BuildDir)) { throw "build dir not found: $BuildDir" }

if (-not $VcVars) {
    $candidates = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending
    if (-not $candidates) { throw "vcvars64.bat not found; pass -VcVars" }
    $VcVars = $candidates[0].FullName
}
if (-not (Test-Path $VcVars)) { throw "vcvars64.bat not found: $VcVars" }
$vcSetup = "call `"$VcVars`"$(if ($VcVersion) { ' -vcvars_ver=' + $VcVersion }) >nul 2>&1"

$cmakeFile = Join-Path $BuildDir "compile_commands.json"
if ($Reconfigure -or -not (Test-Path $cmakeFile) -or
    ($Target -in @('ecm_cuda','ecm_cuda_stage2') -and -not (Test-Path (Join-Path $BuildDir 'cuda_target_graph.json')))) {
    Write-Host "configuring $BuildDir (exporting compile_commands.json) ..."
    $cfg = "$vcSetup && cmake -S `"$repo`" -B `"$BuildDir`" -DCMAKE_BUILD_TYPE=$Config -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    & cmd.exe /c $cfg
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }
    if (-not (Test-Path $cmakeFile)) { throw "cmake did not write compile_commands.json" }
}

$entries = @(Get-EcmTargetCompileEntries $BuildDir $Target) |
    Where-Object { $_.file -match $SourceRegex } |
    Where-Object { -not $Only -or $_.file -match $Only }
if (-not $entries) { throw "no .cu entries matched (-Only '$Only')" }

if ($Jobs -le 0) { $Jobs = [Math]::Min(6, [Environment]::ProcessorCount) }
$Jobs = [Math]::Max(1, [Math]::Min($Jobs, @($entries).Count))

$logDir = Join-Path $BuildDir "par_nvcc"
New-Item -ItemType Directory -Force $logDir | Out-Null

# --- decide what to compile -------------------------------------------------
$tuList = @()
foreach ($e in $entries) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($e.file)
    $obj = Get-EcmObjectPath $e $BuildDir
    if (@($entries | Where-Object { [IO.Path]::GetFileNameWithoutExtension($_.file) -ieq $name }).Count -gt 1) {
        $name = (Get-EcmSourceKey $BuildDir $obj) -replace '[^A-Za-z0-9_.-]', '_'
    }
    $state = Get-EcmCompileState $repo $e $vcSetup
    $reason = if ($SkipUpToDate -and -not $Rebuild) { Get-EcmCompileInvalidationReason $state } else { 'forced rebuild' }
    $skip = $reason -eq ''
    if ($HostOnly) {
        if (-not $skip) { throw "HostOnly CUDA object is not reusable: $($e.file): $reason" }
        # NMake also uses timestamps. Reject a pending native rebuild rather
        # than letting its final host/link step silently invoke nvcc.
        $objectTime = (Get-Item -LiteralPath $obj).LastWriteTimeUtc
        $nativeInputs = @($state.source_hashes.Keys | ForEach-Object { Resolve-EcmSourcePath $repo $_ })
        $owner = [regex]::Match((Get-EcmSourceKey $BuildDir $obj), '^(CMakeFiles/[^/]+\.dir)/')
        if ($owner.Success) {
            $nativeInputs += Join-Path $BuildDir ($owner.Groups[1].Value + '/flags.make')
            $nativeInputs += Join-Path $BuildDir ($owner.Groups[1].Value + '/compiler_depend.ts')
        }
        foreach ($inputPath in $nativeInputs) {
            if ((Test-Path -LiteralPath $inputPath) -and (Get-Item -LiteralPath $inputPath).LastWriteTimeUtc -gt $objectTime) {
                throw "HostOnly native CUDA dependency is newer than its object: $inputPath"
            }
        }
    }
    if (-not $skip) { Write-Host ("  rebuild {0}: {1}" -f $name, $reason) }
    $tuList += [pscustomobject]@{
        Name = $name; Entry = $e; Obj = $obj; Skip = $skip
        State = $state
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
$compileStart = Get-Date

while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($queue.Count -gt 0 -and $running.Count -lt $Jobs) {
        $j = $queue.Dequeue()
        $compileCommand = $j.Entry.command
        if ($j.Entry.file -match '\.cu$' -or $j.Entry.command -match '(?i)(?:^|[\\/" ])nvcc(?:\.exe)?(?:"|\s)') {
            $compileCommand += " --generate-dependencies-with-compile --dependency-output `"$($j.Obj).ecm.d`""
        }
        $bat = "@echo off`r`n$vcSetup`r`nif errorlevel 1 exit /b 1`r`ncd /d `"$($j.Entry.directory)`"`r`n$compileCommand`r`n"
        [System.IO.File]::WriteAllText($j.Bat, $bat, [System.Text.UTF8Encoding]::new($false))
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
                $after = Get-EcmCompileState $repo $j.Entry $vcSetup
                Assert-EcmCompileInputs $j.State $after
                Save-EcmCompileState $after
                Write-Host ("  ok   {0} ({1:N1}s)" -f $j.Name, $j.Seconds)
            }
            if ($objMissing) { Write-Host ("       (no object file at {0})" -f $j.Obj) }
        }
    }
}

foreach ($j in $tuList | Where-Object { $_.Skip }) { Write-Host ("  skip {0} (up to date)" -f $j.Name) }

$longest = 0.0
$slowest = $null
foreach ($j in $tuList) { if ($j.Seconds -gt $longest) { $longest = $j.Seconds; $slowest = $j.Name } }
$wall = ((Get-Date) - $compileStart).TotalSeconds
$sum = ($tuList | Measure-Object -Property Seconds -Sum).Sum
Write-Host ("compiled {0} TU(s): elapsed wall {1:N1}s (sum of concurrent TU elapsed times {2:N1}s)" -f `
    $todo.Count, $wall, $sum)
[ordered]@{jobs=$Jobs;wall_seconds=$wall;tu_elapsed_sum_seconds=$sum;
    units=@($tuList|ForEach-Object{[ordered]@{source=$_.Entry.file;object=$_.Obj;skipped=$_.Skip;started=$_.Start;seconds=$_.Seconds}})
}|ConvertTo-Json -Depth 6|Set-Content (Join-Path $logDir 'timings.json') -Encoding utf8
if ($slowest) {
    Write-Host ("longest TU = {0} ({1:N1}s)" -f $slowest, $longest)
}

if ($failed.Count -gt 0) {
    Write-Host ("{0} translation unit(s) FAILED; see {1}" -f $failed.Count, $logDir)
    exit 1
}

foreach ($j in $tuList) {
    $current = Get-EcmCompileState $repo $j.Entry $vcSetup
    if (-not (Test-EcmCompileState $current)) { throw "Compile dependency/object changed during build: $($j.Entry.file)" }
}

if ($NoBuild) { exit 0 }

# --- hand the rest (host objects + link) to nmake ---------------------------
Write-Host "cmake --build $BuildDir --target $Target (host code + link) ..."
$build = "$vcSetup && cmake --build `"$BuildDir`" --config $Config --target $Target"
& cmd.exe /c $build
if ($LASTEXITCODE -ne 0) { throw "cmake --build failed ($LASTEXITCODE)" }
Write-Host "done"
