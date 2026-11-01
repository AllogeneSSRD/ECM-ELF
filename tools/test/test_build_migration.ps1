# Exercise real migration and promotion with disposable CPU-only build fixtures.
param([Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $repo 'tools/build/internal/cuda_build_profile.ps1')
$out = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $out) { throw 'Use a fresh output directory' }
Initialize-EcmBuildDirectory $out
$fixture = Join-Path $out 'fixture'
[void][IO.Directory]::CreateDirectory($fixture)
& git init --quiet $fixture
if ($LASTEXITCODE) { throw 'Fixture git init failed' }
function Check([bool]$ok,[string]$message) { if (-not $ok) { throw $message } }
foreach ($name in @('build_case [1]','build_cuda_cmake','build_stage1_local','build_gui_v2_local')) {
    $path = Join-Path $fixture $name
    Initialize-EcmBuildDirectory $path
    [IO.File]::WriteAllText((Join-Path $path 'payload.txt'),$name)
}
$tracked = Join-Path $fixture 'build_scripts'
[void][IO.Directory]::CreateDirectory($tracked)
[IO.File]::WriteAllText((Join-Path $tracked 'source.ps1'),'tracked source')
& git -C $fixture add -- build_scripts/source.ps1
if ($LASTEXITCODE) { throw 'Cannot stage fixture source' }
$old = [ordered]@{role='local';stage='stage2';source_root=$fixture;variant='old'}
$new = [ordered]@{role='local';stage='stage2';source_root=$fixture;variant='new'}
$fixed = Join-Path $fixture 'build_stage2_local'
$profile = Join-Path $fixture ('build_stage2_local_'+(Get-EcmProfileHash $new).Substring(0,16))
foreach ($pair in @(@($fixed,$old),@($profile,$new))) {
    Initialize-EcmBuildDirectory $pair[0]
    $marker = [ordered]@{schema=1;fields=$pair[1];profile_sha256=(Get-EcmProfileHash $pair[1])}
    [IO.File]::WriteAllText((Join-Path $pair[0] 'profile.json'),($marker | ConvertTo-Json -Depth 4))
    $exe = Join-Path $pair[0] 'ecm_cuda_stage2.exe'
    [IO.File]::WriteAllText($exe,$pair[1].variant)
    $manifest = @{profile_sha256=$marker.profile_sha256;sha256=(Get-FileHash -LiteralPath $exe).Hash}
    [IO.File]::WriteAllText((Join-Path $pair[0] 'build_manifest.json'),($manifest | ConvertTo-Json))
}
$script = Join-Path $repo 'tools/build/test/move_root_test_builds.ps1'
& $script -Repo $fixture -PromoteStage2 $profile
Check (Test-Path -LiteralPath $profile) 'Dry-run moved the promotion source'
Check ([IO.File]::ReadAllText((Join-Path $fixed 'ecm_cuda_stage2.exe')) -eq 'old') 'Dry-run replaced the current program'
& $script -Repo $fixture -PromoteStage2 $profile -Apply
Check ([IO.File]::ReadAllText((Join-Path $fixed 'ecm_cuda_stage2.exe')) -eq 'new') 'Promotion did not preserve the selected executable'
foreach ($name in @('build_case [1]','build_cuda_cmake')) {
    Check (-not (Test-Path -LiteralPath (Join-Path $fixture $name))) 'Test cache remains at root'
    Check ([IO.File]::ReadAllText((Join-Path $fixture "build_dev/$name/payload.txt")) -eq $name) 'Migration changed cached bytes'
}
foreach ($name in @('build_stage1_local','build_gui_v2_local','build_scripts')) {
    Check (Test-Path -LiteralPath (Join-Path $fixture $name)) 'Formal output or tracked source was moved'
}
$archive = @(Get-ChildItem -LiteralPath (Join-Path $fixture 'build_dev/stage2_local_profiles') -Directory)
Check ($archive.Count -eq 1 -and [IO.File]::ReadAllText((Join-Path $archive[0].FullName 'ecm_cuda_stage2.exe')) -eq 'old') 'Old local build was not retained'
$receipt = Get-ChildItem -LiteralPath (Join-Path $fixture 'build_dev/migrations') -Filter '*.json' |
    ForEach-Object {Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json} | Where-Object apply
Check ($receipt.complete -and @($receipt.moves | Where-Object {-not $_.done}).Count -eq 0) 'Applied receipt is incomplete'
$before = @(& git -C $fixture status --porcelain --untracked-files=all)
Check ($before.Count -eq 1 -and $before[0] -like '*build_scripts/source.ps1') 'Migration exposed ignored artifacts to Git'
$caught = ''
try { Move-EcmBuildDirectory $fixture $fixed (Join-Path $fixture '../escaped') } catch { $caught = $_.Exception.Message }
Check ($caught -like '*inside workspace*') 'Build moves accepted an escaping path'
Check (Test-Path -LiteralPath $fixed) 'Rejected move changed source'
Write-Host 'PASS: migration, promotion, ignore policy, tracked-source retention and path containment'
