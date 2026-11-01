# Move ignored root-level test caches into build_dev without rewriting evidence.
# With no -Apply, only a migration receipt/plan is written; no cache is moved.
param([string]$Repo = '', [string]$PromoteStage2 = '', [switch]$Apply)
$ErrorActionPreference = 'Stop'
if (-not $Repo) { $Repo = Join-Path $PSScriptRoot '../../..' }
$Repo = [IO.Path]::GetFullPath($Repo).TrimEnd('\','/')
. (Join-Path $PSScriptRoot '../internal/cuda_build_profile.ps1')
$container = Join-Path $Repo 'build_dev'
Initialize-EcmBuildDirectory $container
$receiptRoot = Join-Path $container 'migrations'
Initialize-EcmBuildDirectory $receiptRoot
$receipt = Join-Path $receiptRoot ([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')+'_'+[Guid]::NewGuid().ToString('N')+'.json')
$moves = New-Object 'Collections.Generic.List[object]'
$kept = New-Object 'Collections.Generic.List[string]'
$tracked = @(& git -C $Repo ls-files -- ':(glob)build*/**')
if ($LASTEXITCODE) { throw 'Cannot identify tracked build scripts; no migration performed' }
$trackedRoots = @($tracked | ForEach-Object { $_.Split('/')[0] } | Sort-Object -Unique)
$formal = @('build_dev','build_cuda','build_cuda_dev','build_cuda_local','build_cuda_release',
    'build_opencl_local','build_opencl_release','build_cpu_local','build_gui','build_gui_local',
    'build_gui_v2_local','build_stage1_local','build_stage2_local','build_android_local','build_android_release')
$promotion = ''
if ($PromoteStage2) {
    $promotion = if ([IO.Path]::IsPathRooted($PromoteStage2)) { $PromoteStage2 } else { Join-Path $Repo $PromoteStage2 }
    $promotion = [IO.Path]::GetFullPath($promotion).TrimEnd('\','/')
    if ([IO.Path]::GetDirectoryName($promotion) -ine $Repo -or
        [IO.Path]::GetFileName($promotion) -notmatch '^build_stage2_local_[0-9a-f]{16}$') {
        throw 'PromoteStage2 must name one explicit root-level Stage2 local profile directory'
    }
    $marker = Get-Content -LiteralPath (Join-Path $promotion 'profile.json') -Raw | ConvertFrom-Json
    if ($marker.schema -ne 1 -or $marker.fields.role -ne 'local' -or $marker.fields.stage -ne 'stage2' -or
        $marker.fields.source_root -ine $Repo -or (Get-EcmProfileHash $marker.fields) -cne $marker.profile_sha256 -or
        [IO.Path]::GetFileName($promotion) -cne ('build_stage2_local_'+$marker.profile_sha256.Substring(0,16))) {
        throw 'Promotion requires a valid same-workspace local Stage2 profile'
    }
    $manifest = Get-Content -LiteralPath (Join-Path $promotion 'build_manifest.json') -Raw | ConvertFrom-Json
    if ($manifest.profile_sha256 -cne $marker.profile_sha256 -or
        (Get-FileHash -LiteralPath (Join-Path $promotion 'ecm_cuda_stage2.exe')).Hash -ine $manifest.sha256) {
        throw 'Promoted executable differs from its build manifest'
    }
    $fixed = Join-Path $Repo 'build_stage2_local'
    if (Test-Path -LiteralPath $fixed) {
        if ('build_stage2_local' -in $trackedRoots) { throw 'Cannot archive a tracked build directory' }
        $oldMarker = Get-Content -LiteralPath (Join-Path $fixed 'profile.json') -Raw | ConvertFrom-Json
        if ($oldMarker.schema -ne 1 -or $oldMarker.fields.role -ne 'local' -or
            $oldMarker.fields.stage -ne 'stage2' -or $oldMarker.fields.source_root -ine $Repo -or
            (Get-EcmProfileHash $oldMarker.fields) -cne $oldMarker.profile_sha256) {
            throw 'Existing fixed Stage2 directory has an invalid or foreign profile'
        }
        $archive = Join-Path $container ('stage2_local_profiles/'+$oldMarker.profile_sha256.Substring(0,16)+'_'+[Guid]::NewGuid().ToString('N'))
        $moves.Add([pscustomobject]@{source=$fixed;destination=$archive;kind='archive-local';done=$false})
    }
    $moves.Add([pscustomobject]@{source=$promotion;destination=$fixed;kind='promote-local';done=$false})
}
foreach ($directory in Get-ChildItem -LiteralPath $Repo -Directory -Filter 'build*') {
    if ($directory.FullName -ieq $promotion) { continue }
    if ($directory.Name -in $formal -or $directory.Name -like 'build_stage1_local_*' -or
        $directory.Name -in $trackedRoots) { $kept.Add($directory.Name); continue }
    if ($directory.Name -notmatch '^build(?:_|$)') { continue }
    $moves.Add([pscustomobject]@{source=$directory.FullName;destination=(Join-Path $container $directory.Name);kind='test-cache';done=$false})
}
# Validate every absolute path before the first move, including the promotion's
# occupied target (vacated by its earlier archive operation).
$vacated = @{}
foreach ($move in $moves) {
    foreach ($path in @($move.source,$move.destination)) {
        if (-not ([IO.Path]::GetFullPath($path)).StartsWith($Repo+'\',[StringComparison]::OrdinalIgnoreCase)) {
            throw "Migration escaped the workspace: $path"
        }
    }
    if ((Test-Path -LiteralPath $move.destination) -and -not $vacated.ContainsKey($move.destination)) {
        throw "Migration destination already exists: $($move.destination)"
    }
    if ((Get-Item -LiteralPath $move.source -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "Migration cannot move a junction: $($move.source)"
    }
    $vacated[$move.source] = $true
}
$state = [ordered]@{schema=1;workspace=$Repo;apply=[bool]$Apply;complete=$false;kept=$kept.ToArray();moves=$moves.ToArray();error=''}
function Save-Receipt {
    [IO.File]::WriteAllText($receipt, ($state | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
}
Save-Receipt
if ($Apply) {
    # Do not terminate running workers or helpers to make a directory movable.
    foreach ($process in Get-Process) {
        $executable = $null
        try { $executable = $process.Path } catch { continue }
        if (-not $executable) { continue }
        foreach ($move in $moves) {
            if ($executable.StartsWith($move.source+'\',[StringComparison]::OrdinalIgnoreCase)) {
                throw "Build is in use by PID $($process.Id): $executable"
            }
        }
    }
    try {
        foreach ($move in $moves) {
            Move-EcmBuildDirectory $Repo $move.source $move.destination
            $move.done = $true
            Save-Receipt
        }
        $state.complete = $true
    } catch {
        $state.error = $_.Exception.Message
        throw
    } finally { Save-Receipt }
}
Write-Host "Build migration: $($moves.Count) planned, $(@($moves | Where-Object done).Count) moved; receipt: $receipt"
