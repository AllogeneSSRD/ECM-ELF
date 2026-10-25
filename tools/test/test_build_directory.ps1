# CPU-only checks of output-directory policy and Git behavior. No GPU builds.
param([Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $repo 'tools/build/internal/cuda_build_profile.ps1')
$out=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if(Test-Path -LiteralPath $out){throw 'Use a fresh output directory'}
Initialize-EcmBuildDirectory $out
$checks=New-Object 'Collections.Generic.List[string]'
function Check([bool]$Valid,[string]$Name){
    if(-not $Valid){throw "FAIL: $Name"}
    $checks.Add($Name)
}
function Reject([scriptblock]$Action,[string]$Message,[string]$Name){
    $caught='';try{& $Action | Out-Null}catch{$caught=$_.Exception.Message}
    Check ($caught -like "*$Message*") $Name
}
# A real independent repository avoids the enclosing project's data/ rule.
$gitRepo=Join-Path $out 'git fixture'
[void][IO.Directory]::CreateDirectory($gitRepo)
& git init --quiet $gitRepo
if($LASTEXITCODE){throw 'Fixture git init failed'}
$build=Join-Path $gitRepo 'build space [1]'
Initialize-EcmBuildDirectory $build
$ignore=Join-Path $build '.gitignore'
Check ([IO.File]::ReadAllText($ignore) -ceq "*`n") 'fresh directory contains one UTF-8 ignore-all rule'
$artifact=Join-Path $build 'worker.obj'
[IO.File]::WriteAllText($artifact,'existing object')
$before=(Get-FileHash -LiteralPath $artifact).Hash
$timestamp=[DateTime]::UtcNow.AddDays(-2)
[IO.File]::SetLastWriteTimeUtc($ignore,$timestamp)
Initialize-EcmBuildDirectory $build
Check ([IO.File]::GetLastWriteTimeUtc($ignore) -eq $timestamp) 'valid ignore marker is not rewritten on reuse'
[IO.File]::Delete($ignore)
Initialize-EcmBuildDirectory $build
Check ([IO.File]::ReadAllText($ignore) -ceq "*`n") 'reuse repairs a missing ignore marker'
Check ((Get-FileHash -LiteralPath $artifact).Hash -eq $before) 'reuse preserves existing build artifacts'
[IO.File]::WriteAllText($ignore,"*.obj`n!.gitignore`n")
Initialize-EcmBuildDirectory $build
Check ([IO.File]::ReadAllText($ignore) -ceq "*`n") 'partial rules become ignore-all without a marker exception'
& git -C $gitRepo check-ignore --quiet -- 'build space [1]/.gitignore'
Check ($LASTEXITCODE -eq 0) 'Git ignores the marker itself'
& git -C $gitRepo check-ignore --quiet -- 'build space [1]/worker.obj'
Check ($LASTEXITCODE -eq 0) 'Git ignores build artifacts'
$status=& git -C $gitRepo status --porcelain --untracked-files=all
Check (-not $status -and $LASTEXITCODE -eq 0) 'ignored build directory introduces no untracked files'
Reject {Initialize-EcmBuildDirectory $repo} 'roots cannot' 'repository root cannot be ignored'
Reject {Initialize-EcmBuildDirectory ([IO.Path]::GetPathRoot($repo))} 'roots cannot' 'filesystem root cannot be ignored'
Reject {Initialize-EcmBuildDirectory (Join-Path $repo 'tools/build')} 'source directory' 'build script sources cannot be ignored'
Reject {Initialize-EcmBuildDirectory (Join-Path $repo 'src/ecm')} 'source directory' 'program sources cannot be ignored'
$container=Join-Path $out 'build_nested'
$leaf=Join-Path $container 'stage1/profile'
Initialize-EcmBuildDirectory $leaf
Check ([IO.File]::ReadAllText((Join-Path $container '.gitignore')) -ceq "*`n") 'nested profiles protect their build container'
Check ([IO.File]::ReadAllText((Join-Path $leaf '.gitignore')) -ceq "*`n") 'nested profile leaf is also initialized'
$fields=[ordered]@{role='local';stage='stage1';identity='fixture'}
$profile=[pscustomobject]@{Hash=(Get-EcmProfileHash $fields);Fields=$fields}
Assert-EcmBuildProfile $leaf $profile
$marker=Join-Path $leaf 'profile.json'
$hash=(Get-FileHash -LiteralPath $marker).Hash
[IO.File]::Delete((Join-Path $leaf '.gitignore'))
Assert-EcmBuildProfile $leaf $profile -ReadOnly
Check ([IO.File]::Exists((Join-Path $leaf '.gitignore'))) 'matching SkipBuild profile repairs its ignore marker'
Check ((Get-FileHash -LiteralPath $marker).Hash -eq $hash) 'profile identity is preserved when repairing ignore marker'
$otherFields=[ordered]@{role='release';stage='stage1';identity='fixture'}
$other=[pscustomobject]@{Hash=(Get-EcmProfileHash $otherFields);Fields=$otherFields}
Reject {Assert-EcmBuildProfile $leaf $other} 'profile/role/engine differs' 'ignore marker does not permit incompatible profile reuse'
$unmarked=Join-Path $out 'unmarked'
Initialize-EcmBuildDirectory $unmarked
Reject {Assert-EcmBuildProfile $unmarked $profile -ReadOnly} 'matching profile marker' 'ignore-only directory does not satisfy SkipBuild'
[IO.File]::WriteAllText((Join-Path $unmarked 'unknown.obj'),'unregistered')
Reject {Assert-EcmBuildProfile $unmarked $profile} 'no matching formal manifest' 'ignore marker does not bypass legacy object validation'
$result=[ordered]@{complete=$true;checks=@($checks);powershell=$PSVersionTable.PSVersion.ToString()}
[IO.File]::WriteAllText((Join-Path $out 'result.json'),($result|ConvertTo-Json -Depth 4),[Text.UTF8Encoding]::new($false))
Write-Host "PASS $($checks.Count) build directory checks"
