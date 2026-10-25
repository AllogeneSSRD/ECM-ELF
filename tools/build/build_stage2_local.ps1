#Requires -Version 5.1
<#
.SYNOPSIS
Build the standalone CUDA Stage2 save/queue executable (MSVC + nvcc + GMP).
.DESCRIPTION
Local builds default to build_stage2_local. Build overrides the final directory;
BuildRoot explicitly retains the profile hierarchy under the requested root.
Incompatible profiles in a fixed directory are refused, never overwritten.
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_stage2_local.ps1 -Arch sm_89
#>
param(
    [string]$Build = '',
    [string]$BuildRoot = '',
    [ValidateSet('local','dev','release')][string]$BuildRole = 'local',
    [ValidateSet('production','development')][string]$Engine = 'production',
    [ValidateSet('grouped','isolated')][string]$KernelLayout = 'grouped',
    [ValidatePattern('^sm_[0-9]+$')][string]$Arch = 'sm_89',
    [ValidatePattern('^$|^[0-9]+$')][string]$PtxArch = '',
    [ValidateSet('runtime','fold','short','ptx')][string]$GlBackend = 'ptx',
    [ValidateSet(0,4)][int]$OuterUnrollU = 0,
    [ValidateSet(0,1,2,3)][int]$AddSubMask = 0,
    [ValidateRange(1,64)][int]$SplitCompile = 8,
    [ValidateRange(1,64)][int]$VerifyJobs=4,
    [ValidateRange(1,64)][int]$Jobs = 8,
    [string]$Gmp = '',
    [string]$CudaRoot = '',
    [string]$VcVars = '',
    [switch]$HostOnly,
    [string]$ResultFile = '',
    [switch]$Rebuild
)
$ErrorActionPreference = 'Stop'
if ($HostOnly -and $Rebuild) { throw 'HostOnly and Rebuild are mutually exclusive' }
if (-not $PSBoundParameters.ContainsKey('AddSubMask') -and $Engine -eq 'production') { $AddSubMask = 1 }
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$buildTimer = [Diagnostics.Stopwatch]::StartNew()
. (Join-Path $PSScriptRoot 'internal/gmp_dependency.ps1')
$gmpDependency = Resolve-EcmGmp -Repo $repo -Gmp $Gmp -Release:($BuildRole -eq 'release')
$Gmp = $gmpDependency.Root
. (Join-Path $PSScriptRoot 'internal/cuda_toolchain.ps1')
$cuda = Resolve-EcmCudaToolkit -Archs $Arch -CudaRoot $CudaRoot
$vc = Resolve-EcmVcEnvironment $cuda -VcVars $VcVars
. (Join-Path $PSScriptRoot 'internal/cuda_build_profile.ps1')
$buildProfile = New-EcmStage2Profile $repo $BuildRole $Engine $Arch $GlBackend $OuterUnrollU $AddSubMask $SplitCompile $cuda $vc $gmpDependency $PtxArch $KernelLayout
$Build = Resolve-EcmBuildDirectory $repo $buildProfile $Build $BuildRoot
Assert-EcmBuildProfile $Build $buildProfile
Write-Host '== Stage2 build =='
Write-Host "   engine    : $Engine"
Write-Host "   GPU       : $Arch"
Write-Host "   NTT       : backend=$GlBackend add/sub=$AddSubMask outer_unroll=$OuterUnrollU"
Write-Host "   parallel  : nvcc split compile=$SplitCompile"
Write-Host "   modules   : $KernelLayout"
Write-Host "   build dir : $Build"
Write-Host "   role/key  : $BuildRole / $($buildProfile.Key)"
Write-Host "   GMP       : $Gmp"
Write-Host "   CUDA      : $($cuda.Root)"
Write-Host "   MSVC      : $($vc.Version)"
& cmake "-DECM_CONFIG_ROOT=$repo" -P (Join-Path $PSScriptRoot 'internal/check_ecm_config.cmake')
if ($LASTEXITCODE -ne 0) { throw 'Generated ECM config is stale; run python tools/gen/generate_ecm_config.py' }
. (Join-Path $PSScriptRoot 'internal/cuda_build_state.ps1')
. (Join-Path $PSScriptRoot 'internal/cuda_timing.ps1')
$inventory = Get-EcmCudaInventory $repo (Join-Path $Build 'cuda_sources.json')
$moduleDirectory = ''
if ($KernelLayout -eq 'isolated') {
    $moduleDirectory = Join-Path $Build 'cuda_modules/stage2'
    & python (Join-Path $PSScriptRoot 'internal/generate_cuda_modules.py') --out $moduleDirectory
    if ($LASTEXITCODE) { throw 'CUDA module generation failed' }
}
$includeDirs = @((Join-Path $Gmp 'include'), 'include','src','src/ecm/backend/cuda')
$sourceSet = Get-EcmStage2SourceSet $repo $inventory $Engine $includeDirs $moduleDirectory
$sources = @($sourceSet.Sources)
$cudaSources = @($sourceSet.CudaSources)
$deps = @($sourceSet.Dependencies)
$cudaDeps = @($sourceSet.CudaDependencies)
$objectNames = Get-EcmObjectNames $sources
$objDir = Join-Path $Build '_objects'
. (Join-Path $PSScriptRoot 'internal/build_directory.ps1')
Initialize-EcmBuildDirectory $objDir
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
$compileEntries = [ordered]@{}
$beforeStates = [ordered]@{}
foreach ($src in $sources) {
    $stem = $objectNames[$src]
    $obj = Join-Path $objDir "$stem.obj"
    $command = if ($src -in $cudaSources) {
        $buildProfile.CompilePrefix + " -c `"$src`" -o `"$obj`""
    } else {
        $dependencyFlag = if ([version]$vc.Version -lt [version]'14.28') { '/showIncludes' } else { "/sourceDependencies `"$obj.ecm-msvc.json`"" }
        $buildProfile.HostCompilePrefix + " /c `"$src`" /Fo`"$obj`" $dependencyFlag"
    }
    $entry = [pscustomobject]@{ file = (Resolve-EcmSourcePath $repo $src); directory = $repo; command = $command; output = $obj }
    $compileEntries[$src] = $entry

}
$requests=@($sources | ForEach-Object {[pscustomobject]@{Entry=$compileEntries[$_]}})
$preflight=@(Invoke-EcmCompileVerification $repo $requests -Setup $vc.Setup -IncludeDirs $includeDirs -Production:($Engine -eq 'production') -VerifyJobs $VerifyJobs -Phase 'Stage2 reuse preflight')
$reuseReasons=@{}
for($i=0;$i -lt $sources.Count;$i++){$beforeStates[$sources[$i]]=$preflight[$i].State;$reuseReasons[$sources[$i]]=$preflight[$i].Reason}
$previous = $null
$manifestPath = Join-Path $Build 'build_manifest.json'
if (Test-Path -LiteralPath $manifestPath) { $previous = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json }
if ($HostOnly) {
    if (-not $previous -or $previous.engine -ne $Engine -or $previous.architecture -ne $Arch -or
        $previous.gl_fixed_mode -ne $glMode -or $previous.outer_unroll_u -ne $OuterUnrollU -or
        $previous.add_sub_mask -ne $AddSubMask -or $previous.split_compile -ne $SplitCompile -or
        $previous.toolkit -ne $cuda.VersionText -or $previous.cuda_root -ne $cuda.Root -or
        $previous.host_compiler -ne $vc.Compiler -or $previous.host_version -ne $vc.Version) {
        throw 'HostOnly requires identical CUDA architecture, backend, schedule and toolkit'
    }
    if (-not $previous.gmp_hashes -or $previous.gmp_hashes.'include/gmp.h' -ne $gmpHashes['include/gmp.h']) {
        throw 'HostOnly requires the same recorded GMP header; rebuild old manifests or changed GMP headers normally'
    }
    foreach ($dep in $cudaDeps) {
        if ($previous.source_hashes.$dep -ne $sourceHashes[$dep]) { throw "HostOnly CUDA dependency changed: $dep" }
    }
    if ($previous.identity_schema -ne 1 -or @($previous.cuda_sources).Count -ne $cudaSources.Count) {
        throw 'HostOnly requires current CUDA dependency records; rebuild old manifests'
    }
    foreach ($src in $cudaSources) {
        if ($src -notin $previous.cuda_sources) { throw "HostOnly CUDA source set changed: $src" }
        $unit = $previous.compile_units.$src
        $state = $beforeStates[$src]
        if (-not $unit -or $unit.fingerprint -ne $state.fingerprint) { throw "HostOnly CUDA compile dependency changed: $src" }
        if ($unit.object_sha256 -ne (Get-FileHash -LiteralPath $state.object).Hash) { throw "HostOnly CUDA object changed: $src" }
    }
}
$fresh = -not $Rebuild -and (Test-Path $exe) -and (Test-Path $signaturePath) -and $previous -and
    ([IO.File]::ReadAllText((Resolve-Path $signaturePath)) -eq $signatureText) -and
    ($previous.sha256 -eq (Get-FileHash -LiteralPath $exe).Hash)
if ($fresh) {
    try { Assert-EcmBuildUnits $repo $Build $previous $sources $includeDirs -Production:($Engine -eq 'production') -VerifyJobs $VerifyJobs }
    catch { $fresh = $false }
}
$linkInputs = if ($previous) { $previous.link_inputs } else { $null }
if (-not $fresh) {
    $objects = @()
    $compiled = 0
    $pendingJobs = @()
    $pendingTimings = @{}
    $compileSeconds = 0.0
    Write-Host "== compile $($sources.Count) translation units =="
    foreach ($src in $sources) {
        $stem = $objectNames[$src]
        $obj = Join-Path $objDir "$stem.obj"
        $log = Join-Path $objDir "$stem.log"
        $objects += $obj
        # The whole-build signature determines whether metadata/linking needs
        # attention, never whether every translation unit must be recompiled.
        # Keep HostOnly's complete CUDA preflight above, before any host writes.
        if ($HostOnly -and $src -in $cudaSources) {
            Write-Host ("  skip {0} (verified CUDA object)" -f $stem)
            continue
        }
        $reason = if ($Rebuild) { 'forced rebuild' } else { $reuseReasons[$src] }
        if (-not $reason) {
            Write-Host ("  skip {0} (verified command, dependencies and object)" -f $stem)
            continue
        }
        $compiled++
        Write-Host ("  start [{0}/{1}] {2}" -f $compiled, ($sources.Count - $(if ($HostOnly) { $cudaSources.Count } else { 0 })), $src)
        Write-Host ("       reason: {0}" -f $reason)
        $compileCommand = $compileEntries[$src].command
        $phaseTiming = $null
        if ($src -in $cudaSources) {
            $compileCommand += " --generate-dependencies-with-compile --dependency-output `"$obj.ecm.d`""
            $phaseTiming = New-EcmNvccTiming $Build $src $compileCommand
            $compileCommand = $phaseTiming.ActualCommand
        }
        $sw = [Diagnostics.Stopwatch]::StartNew()
        # Keep setup and compilation on separate lines. Capture via a pipe:
        # direct cmd file redirection crashes NVCC's compiler probe in some
        # restricted Windows execution environments, even on an empty source.
        $commandFile = Join-Path $objDir "$stem.compile.cmd"
        [IO.File]::WriteAllText($commandFile, "@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$compileCommand`r`n", [Text.UTF8Encoding]::new($false))
        $pendingJobs += [ordered]@{ source=$src; command_file=$commandFile; log=$log
            msvc_includes=$(if ($compileCommand -match '/showIncludes') { "$obj.ecm-msvc.json" } else { $null }) }
        if ($phaseTiming) { $pendingTimings[$src] = $phaseTiming }
    }
    if ($pendingJobs.Count) {
        $jobPlan = Join-Path $objDir 'compile-jobs.json'
        $jobResult = Join-Path $objDir 'compile-results.json'
        ConvertTo-Json -InputObject @($pendingJobs) -Depth 4 | Set-Content -LiteralPath $jobPlan -Encoding UTF8
        & python (Join-Path $PSScriptRoot 'internal/run_compile_jobs.py') --plan $jobPlan --out $jobResult --jobs $Jobs
        $compileExit = $LASTEXITCODE
        if (-not (Test-Path -LiteralPath $jobResult)) { throw 'Compiler executor did not finish; inspect unit logs' }
        $results = Get-Content -LiteralPath $jobResult -Raw | ConvertFrom-Json
        $compileSeconds = $results.wall_seconds
        foreach ($unitResult in $results.units) {
            $src = $unitResult.source
            if ($pendingTimings.ContainsKey($src)) {
                $pendingTimings[$src].Started = [DateTime]::Parse($unitResult.started_utc)
                Complete-EcmNvccTiming $pendingTimings[$src] $unitResult.exit_code $unitResult.elapsed_seconds
            }
            if ($unitResult.exit_code) { Get-Content -LiteralPath $unitResult.log -Tail 25 }
            else {
                $after = Get-EcmCompileState $repo $compileEntries[$src] $vc.Setup $includeDirs -Production:($Engine -eq 'production')
                Assert-EcmCompileInputs $beforeStates[$src] $after
                Save-EcmCompileState $after
            }
        }
        if ($compileExit) { throw 'Stage2 compilation failed; inspect compile-results.json and unit logs' }
    }
    Write-Host ("compiled {0} TU(s): wall {1:N1}s (nvcc split compile={2})" -f $compiled, $compileSeconds, $SplitCompile)
    $objArgs = ($objects | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $linkLog = Join-Path $objDir 'link.log'
    $line = $buildProfile.LinkPrefix + " $objArgs -L `"$Gmp/lib`" -lgmp -o `"$exe`""
    $linkObjects = [ordered]@{}
    foreach ($object in $objects) { $linkObjects[$object] = (Get-FileHash -LiteralPath $object).Hash }
    $linkInputs = [ordered]@{ command = $line; library = $gmpHashes['lib/gmp.lib']; objects = $linkObjects }
    $reuseLink = -not $Rebuild -and $previous -and $previous.link_inputs -and
        (Test-Path -LiteralPath $exe) -and $previous.sha256 -eq (Get-FileHash -LiteralPath $exe).Hash -and
        $previous.link_inputs.command -eq $line -and $previous.link_inputs.library -eq $linkInputs.library
    if ($reuseLink) {
        try { Assert-EcmHashes $previous.link_inputs.objects $linkObjects 'Link object changed' }
        catch { $reuseLink = $false }
    }
    if (-not $reuseLink) {
        Write-Host '== link ecm_cuda_stage2.exe =='
        $linkTimer = [Diagnostics.Stopwatch]::StartNew()
        $commandFile = Join-Path $objDir 'link.cmd'
        $linkCommand = $line.Substring(($vc.Setup + ' && ').Length)
        $linkTiming = New-EcmNvccTiming $Build 'stage2-link' $linkCommand
        $linkCommand = $linkTiming.ActualCommand
        [IO.File]::WriteAllText($commandFile, "@echo off`r`n$($vc.Setup)`r`nif errorlevel 1 exit /b 1`r`n$linkCommand`r`n", [Text.UTF8Encoding]::new($false))
        & cmd.exe /d /c $commandFile 2>&1 | Out-File -LiteralPath $linkLog -Encoding utf8
        Complete-EcmNvccTiming $linkTiming $LASTEXITCODE
        if ($LASTEXITCODE -ne 0) { Get-Content $linkLog -Tail 40; throw 'link failed' }
        $linkTimer.Stop()
        Write-Host ("  ok   link ({0:N1}s)" -f $linkTimer.Elapsed.TotalSeconds)
    } else { Write-Host 'Link command, library, objects and executable verified; link skipped.' }
    foreach ($dep in $deps) {
        if ((Get-FileHash -LiteralPath $dep -Algorithm SHA256).Hash -ne $sourceHashes[$dep]) {
            throw "Source changed during build: $dep"
        }
    }
    $currentSet = Get-EcmStage2SourceSet $repo $inventory $Engine $includeDirs $moduleDirectory
    Assert-EcmHashes $sourceHashes (Get-EcmSourceHashes $repo $currentSet.Dependencies) 'Source changed during build'
    if ($HostOnly) {
        foreach ($src in $cudaSources) {
            if ((Get-FileHash $beforeStates[$src].object).Hash -ne $previous.compile_units.$src.object_sha256) { throw "Reused CUDA object changed during host build: $src" }
        }
    }
    [IO.File]::WriteAllText((Join-Path (Resolve-Path $objDir) 'build_signature.txt'), $signatureText)
} else { Write-Host 'Source/toolkit/architecture signature unchanged; executable reused.' }
foreach ($name in $gmpHashes.Keys) {
    if ((Get-FileHash -LiteralPath (Join-Path $Gmp $name)).Hash -ne $gmpHashes[$name]) { throw "GMP changed during build: $name" }
}
Copy-Item -LiteralPath (Join-Path $Gmp 'bin/gmp-10.dll') -Destination $Build -Force
# The queue driver is built without CMake; build the same diagnostic host target.
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'test/build_diagnostics.ps1') -Build (Join-Path $Build 'diagnostics-build') -GmpRoot $Gmp -VcVars $vc.VcVars
if ($LASTEXITCODE) { throw 'Stage2 diagnostic host build failed' }
$diagnosticFiles = & python (Join-Path $repo 'tools/packaging/diagnostics_runtime.py') --package-build (Join-Path $Build 'diagnostics-build') --out $Build
if ($LASTEXITCODE) { throw 'Stage2 diagnostic runtime deployment failed' }
Copy-Item -LiteralPath (Join-Path $Build 'diagnostics-build/diagnostics_build_manifest.json') -Destination $Build -Force

$objectHashes = [ordered]@{}
$compileUnits = [ordered]@{}
$postcheck=@(Invoke-EcmCompileVerification $repo $requests -Setup $vc.Setup -IncludeDirs $includeDirs -Production:($Engine -eq 'production') -ManifestOnly -VerifyJobs $VerifyJobs -Phase 'Stage2 postcheck')
$unitIndex=0
foreach ($src in $sources) {
    $stem = $objectNames[$src]
    $objectHash = $postcheck[$unitIndex].ObjectHash
    $objectHashes[$stem] = $objectHash
    $state = $postcheck[$unitIndex].State;$unitIndex++
    if ($fresh -or ($HostOnly -and $src -in $cudaSources)) {
        $expected = $previous.compile_units.$src.fingerprint
        $expectedObject = $previous.compile_units.$src.object_sha256
    } else {
        $record = Get-Content -LiteralPath "$($state.object).ecm-state.json" -Raw | ConvertFrom-Json
        $expected = $record.fingerprint
        $expectedObject = $record.object_sha256
    }
    if ($expected -ne $state.fingerprint) { throw "Compile dependency changed during build: $src" }
    if ($expectedObject -ne $objectHash) { throw "Compile object changed during build: $src" }
    $compileUnits[$src] = [ordered]@{ entry = $compileEntries[$src]; fingerprint = $state.fingerprint; object_sha256 = $objectHash }
}
$manifest = [ordered]@{
    identity_schema = 1
    role = $BuildRole
    profile_sha256 = $buildProfile.Hash
    source_inventory_sha256 = (Get-FileHash -LiteralPath (Join-Path $repo 'tools/build/internal/cuda_sources.cmake')).Hash
    compile_setup = $vc.Setup
    compile_units = $compileUnits
    link_inputs = $linkInputs
    cuda_sources = $cudaSources
    exe = (Resolve-Path $exe).Path
    sha256 = (Get-FileHash $exe -Algorithm SHA256).Hash
    architecture = $Arch
    ptx_architecture = $PtxArch
    module_directory = $moduleDirectory
    kernel_layout = $KernelLayout
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
    host_driver = 'msvc'
    sources = $signature
    source_hashes = $sourceHashes
    objects = $objectHashes
    host_only = [bool]$HostOnly
    gmp_root = $Gmp
    gmp_hashes = $gmpHashes
}
Assert-EcmBuildProfile $Build $buildProfile -ReadOnly
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $Build 'build_manifest.json')
$buildTimer.Stop()
Write-Host '== produced =='
Write-Host ("   ecm_cuda_stage2.exe: {0:N0} B ({1:N1} MB)" -f (Get-Item $exe).Length, ((Get-Item $exe).Length / 1MB))
Write-Host "   executable : $exe"
Write-Host "   logs       : $objDir"
Write-Host ("   elapsed    : {0:N1}s" -f $buildTimer.Elapsed.TotalSeconds)
Write-Host 'done'
if ($ResultFile) {
    [ordered]@{ stage = 'stage2'; build = $Build; manifest = $manifestPath; exe = $exe } |
        ConvertTo-Json | Set-Content -LiteralPath $ResultFile -Encoding UTF8
}
exit 0
