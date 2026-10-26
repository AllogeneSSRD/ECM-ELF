# CPU-only profile identity, isolation and marker mutation gates. No compilers.
param([Parameter(Mandatory=$true)][string]$Output, [string]$AlternatePowerShell = '')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $repo 'tools/build/internal/cuda_build_profile.ps1')
$out = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if (Test-Path -LiteralPath $out) { throw 'Use a fresh output directory' }
New-Item -ItemType Directory -Path $out | Out-Null
$checks = New-Object 'System.Collections.Generic.List[string]'
function Check([bool]$Valid, [string]$Name) {
    if (-not $Valid) { throw "FAIL: $Name" }
    $checks.Add($Name)
}
function Reject([scriptblock]$Action, [string]$Message, [string]$Name) {
    $caught = ''
    try { & $Action | Out-Null } catch { $caught = $_.Exception.Message }
    Check ($caught -like "*$Message*") $Name
}
function Write-Fixture([string]$Path, [string]$Text) { [IO.File]::WriteAllText($Path, $Text, [Text.UTF8Encoding]::new($false)) }
$fixture = Join-Path $out 'fixture'
New-Item -ItemType Directory -Path $fixture | Out-Null
$nvcc = Join-Path $fixture 'nvcc.exe'; $cl = Join-Path $fixture 'cl.exe'
Write-Fixture $nvcc 'CUDA compiler identity'
Write-Fixture $cl 'MSVC compiler identity'
$cuda = [pscustomobject]@{Root=$fixture;Nvcc=$nvcc;VersionText='Cuda compilation tools, release 13.3, V13.3.73'}
$vc = [pscustomobject]@{Compiler=$cl;Version='14.51.36231';Setup='call fake-vcvars >nul 2>&1'}
$gmp = [pscustomobject]@{Root=(Join-Path $fixture 'gmp');Hashes=[ordered]@{'include/gmp.h'='header';'lib/gmp.lib'='library';'bin/gmp-10.dll'='runtime'}}
function Profile([string]$Role='local',[string]$Engine='production',[int]$Split=8,[string]$Arch='sm_89') {
    New-EcmStage2Profile $fixture $Role $Engine $Arch ptx 0 1 $Split $cuda $vc $gmp
}
$a = Profile
Check ($a.Key -match '^[0-9a-f]{16}$' -and $a.Hash.Length -eq 64) 'short directory key and full configuration identity'
$leaf = Resolve-EcmBuildDirectory $fixture $a
Check ($leaf -eq (Join-Path $fixture 'build_stage2_local')) 'local Stage2 has a fixed short default directory'
Check ((Resolve-EcmBuildDirectory $fixture $a '' 'profiles').Replace('\','/').EndsWith("profiles/stage2/production/sm89/$($a.Key)")) 'explicit build root retains profile hierarchy'
Write-Fixture (Join-Path $fixture 'business.cpp') 'source version 1'
$same = Profile
Write-Fixture (Join-Path $fixture 'business.cpp') 'source version 2'
Check ((Profile).Hash -eq $same.Hash) 'business source changes do not create a new profile'
Check ((Profile -Split 4).Hash -ne $a.Hash) 'compile schedule changes select another profile'
Check ((Profile -Arch sm_86).Hash -ne $a.Hash) 'architecture changes select another profile'
$dev = Profile -Role dev
Check ($dev.Hash -ne $a.Hash -and (Resolve-EcmBuildDirectory $fixture $dev) -like '*build_cuda_dev*') 'development and local roles remain isolated'
$experimental = Profile -Role dev -Engine development
Check ((Resolve-EcmBuildDirectory $fixture $experimental) -like '*stage2\experimental\sm89*') 'experimental engine stays explicitly separate'
$release = Profile -Role release
Check ((Resolve-EcmBuildDirectory $fixture $release) -like '*build_cuda_release\stage2\cuda13_3\sm89*') 'release is grouped by toolkit and architecture'
$isolated = New-EcmStage2Profile $fixture release production sm_89 ptx 0 1 1 $cuda $vc $gmp '' isolated
$grouped = New-EcmStage2Profile $fixture release production sm_89 ptx 0 1 1 $cuda $vc $gmp '' grouped
Check ($isolated.Hash -ne $grouped.Hash) 'Stage2 grouped and isolated layouts cannot share a profile'
Check ($isolated.CompilePrefix -match 'ECM_CUDA_ISOLATED_DEVICE' -and $grouped.CompilePrefix -notmatch 'ECM_CUDA_ISOLATED_DEVICE') 'isolated constants and dispatch are enabled only by explicit layout'
Reject { New-EcmStage2Profile $fixture dev development sm_89 ptx 0 1 8 $cuda $vc $gmp '' isolated } 'production engine' 'isolated generator cannot silently replace a development engine'
Reject { Profile -Role release -Engine development } 'production engine' 'experimental engine cannot become release'
$explicit = Join-Path $fixture 'explicit_leaf'
Check ((Resolve-EcmBuildDirectory $fixture $a $explicit) -eq $explicit) 'explicit Build remains a final leaf without appended profile'
Reject { Resolve-EcmBuildDirectory $fixture $dev $leaf } 'another role' 'local path cannot be reused as dev'
Reject { Resolve-EcmBuildDirectory $fixture $a $fixture } 'roots cannot' 'repository root cannot become a build leaf'
Reject { Resolve-EcmBuildDirectory $fixture $a $explicit 'custom_root' } 'not both' 'leaf and root arguments are mutually exclusive'
Assert-EcmBuildProfile $leaf $a
Assert-EcmBuildProfile $leaf $same -ReadOnly
Check $true 'matching profile marker is reusable and read-only verification succeeds'
Reject { Assert-EcmBuildProfile $leaf (Profile -Split 4) } 'profile/role/engine differs' 'mismatched profile cannot relabel existing objects'
$marker = Join-Path $leaf 'profile.json'; $original = [IO.File]::ReadAllBytes($marker)
$json = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
$json.fields.role = 'dev'
Write-Fixture $marker ($json | ConvertTo-Json -Depth 5)
Reject { Assert-EcmBuildProfile $leaf $a } 'profile/role/engine differs' 'tampered marker fields fail even if full hash is retained'
[IO.File]::WriteAllBytes($marker,$original)
$unmarked = Join-Path $fixture 'unmarked'; New-Item -ItemType Directory -Path $unmarked | Out-Null
Write-Fixture (Join-Path $unmarked 'unknown.obj') 'unknown artifact'
Reject { Assert-EcmBuildProfile $unmarked $a } 'no matching formal manifest' 'unmarked unknown outputs cannot be silently adopted'
Reject { Assert-EcmBuildProfile (Join-Path $fixture 'missing') $a -ReadOnly } 'required for SkipBuild' 'SkipBuild cannot create a new profile marker'
$stage1Options = [ordered]@{configure='-DCMAKE_BUILD_TYPE=Release -DECM_TIERS=384 -DECM_CUDA_ARCHITECTURES=89';tiers='384';enable_prac='0';extra=''}
$stage1 = New-EcmCudaProfile $fixture local stage1 production 89 $cuda $vc $gmp $stage1Options
Check ((Resolve-EcmBuildDirectory $fixture $stage1) -eq (Join-Path $fixture 'build_stage1_local')) 'local Stage1 has a separate fixed short default directory'
Reject { Resolve-EcmBuildDirectory $fixture $stage1 $leaf } 'another stage' 'fixed Stage2 directory cannot be reused for Stage1'
# Reproduce a default-policy update over a valid pre-isolation local cache.
$oldStage1Leaf = Resolve-EcmBuildDirectory $fixture $stage1
Assert-EcmBuildProfile $oldStage1Leaf $stage1
$oldMarkerHash = (Get-FileHash -LiteralPath (Join-Path $oldStage1Leaf 'profile.json')).Hash
$oldObject = Join-Path $oldStage1Leaf 'cached.obj'; Write-Fixture $oldObject 'old grouped object'
$oldObjectHash = (Get-FileHash -LiteralPath $oldObject).Hash
$splitOptions = [ordered]@{configure=($stage1Options.configure+' -DECM_CUDA_ISOLATE_TIERS=ON');tiers='384';enable_prac='0';extra=''}
$splitStage1 = New-EcmCudaProfile $fixture local stage1 production 89 $cuda $vc $gmp $splitOptions
$splitLeaf = Resolve-EcmBuildDirectory $fixture $splitStage1
Check ($splitLeaf -eq ($oldStage1Leaf+'_'+$splitStage1.Key)) 'changed default Stage1 policy selects a deterministic separate directory'
Assert-EcmBuildProfile $splitLeaf $splitStage1
Check ((Resolve-EcmBuildDirectory $fixture $splitStage1) -eq $splitLeaf) 'repeated default Stage1 build reuses its matching alternate profile'
Assert-EcmBuildProfile $splitLeaf $splitStage1 -ReadOnly
Check ((Get-FileHash -LiteralPath (Join-Path $oldStage1Leaf 'profile.json')).Hash -eq $oldMarkerHash -and
    (Get-FileHash -LiteralPath $oldObject).Hash -eq $oldObjectHash) 'automatic selection preserves old marker and objects'
Check ((Resolve-EcmBuildDirectory $fixture $splitStage1 $oldStage1Leaf) -eq $oldStage1Leaf) 'explicit Stage1 BuildDir never redirects'
Reject { Assert-EcmBuildProfile $oldStage1Leaf $splitStage1 } 'option/configure' 'explicit profile mismatch reports the changed configuration field'
$alternateStage2 = Profile -Split 4
$alternateLeaf = Resolve-EcmBuildDirectory $fixture $alternateStage2
Check ($alternateLeaf -eq ($leaf+'_'+$alternateStage2.Key)) 'Stage2 default policy changes also select a separate profile'
Assert-EcmBuildProfile $alternateLeaf $alternateStage2
Check ((Resolve-EcmBuildDirectory $fixture $a) -eq $leaf) 'matching original Stage2 profile keeps its short directory'
$damaged = Get-Content -LiteralPath (Join-Path $oldStage1Leaf 'profile.json') -Raw | ConvertFrom-Json
$damaged.fields.role = 'dev'
Write-Fixture (Join-Path $oldStage1Leaf 'profile.json') ($damaged|ConvertTo-Json -Depth 5)
Check ((Resolve-EcmBuildDirectory $fixture $splitStage1) -eq $oldStage1Leaf) 'corrupt default marker cannot be hidden by automatic redirection'
Reject { Assert-EcmBuildProfile $oldStage1Leaf $splitStage1 } 'profile/role/engine differs' 'corrupt default profile still fails closed'
Write-Fixture (Join-Path $oldStage1Leaf 'profile.json') ([ordered]@{schema=1;profile_sha256=$stage1.Hash;fields=$stage1.Fields}|ConvertTo-Json -Depth 5)
$foreign = New-EcmCudaProfile $fixture dev stage1 production 89 $cuda $vc $gmp $stage1Options
Write-Fixture (Join-Path $oldStage1Leaf 'profile.json') ([ordered]@{schema=1;profile_sha256=$foreign.Hash;fields=$foreign.Fields}|ConvertTo-Json -Depth 5)
Check ((Resolve-EcmBuildDirectory $fixture $splitStage1) -eq $oldStage1Leaf) 'valid foreign-role marker cannot redirect a local default'
Reject { Assert-EcmBuildProfile $oldStage1Leaf $splitStage1 } 'role' 'foreign-role default is refused without replacing its marker'
Write-Fixture (Join-Path $oldStage1Leaf 'profile.json') ([ordered]@{schema=1;profile_sha256=$stage1.Hash;fields=$stage1.Fields}|ConvertTo-Json -Depth 5)
Write-Fixture (Join-Path $splitLeaf 'profile.json') ([ordered]@{schema=1;profile_sha256=$stage1.Hash;fields=$stage1.Fields}|ConvertTo-Json -Depth 5)
Check ((Resolve-EcmBuildDirectory $fixture $splitStage1) -eq $splitLeaf) 'alternate directory is deterministic even when occupied'
Reject { Assert-EcmBuildProfile $splitLeaf $splitStage1 } 'option/configure' 'incompatible alternate cache is refused rather than overwritten'
Write-Fixture (Join-Path $splitLeaf 'profile.json') ([ordered]@{schema=1;profile_sha256=$splitStage1.Hash;fields=$splitStage1.Fields}|ConvertTo-Json -Depth 5)
$legacy = Join-Path $fixture 'legacy_stage1'; New-Item -ItemType Directory -Path $legacy | Out-Null
$legacyManifest = [ordered]@{identity_schema=1;architecture='89';tiers='384';enable_prac=$false
    cuda_root=$cuda.Root;toolkit=$cuda.VersionText;host_compiler=$vc.Compiler;host_version=$vc.Version
    dependency_hashes=[ordered]@{header='header';library='library';runtime='runtime'}}
Write-Fixture (Join-Path $legacy 'stage1_local_manifest.json') ($legacyManifest | ConvertTo-Json -Depth 4)
$versionParts = $stage1.Fields.cmake_version.Split('.')
$cache = "CMAKE_CACHE_MAJOR_VERSION:INTERNAL=$($versionParts[0])`nCMAKE_CACHE_MINOR_VERSION:INTERNAL=$($versionParts[1])`nCMAKE_CACHE_PATCH_VERSION:INTERNAL=$($versionParts[2])`n" +
    "CMAKE_BUILD_TYPE:STRING=Release`nECM_TIERS:STRING=384`nECM_CUDA_ARCHITECTURES:STRING=89`n"
Write-Fixture (Join-Path $legacy 'CMakeCache.txt') $cache
Assert-EcmBuildProfile $legacy $stage1
Assert-EcmBuildProfile $legacy $stage1 -ReadOnly
Check $true 'compatible formal legacy Stage1 cache is marked in place without moving objects'
$wrongLegacy = Join-Path $fixture 'legacy_mismatch'; New-Item -ItemType Directory -Path $wrongLegacy | Out-Null
Write-Fixture (Join-Path $wrongLegacy 'stage1_local_manifest.json') ($legacyManifest | ConvertTo-Json -Depth 4)
Write-Fixture (Join-Path $wrongLegacy 'CMakeCache.txt') $cache
$stage1Options['tiers']='512'; $stage1Options['configure']=$stage1Options['configure'].Replace('TIERS=384','TIERS=512')
$wrongStage1 = New-EcmCudaProfile $fixture local stage1 production 89 $cuda $vc $gmp $stage1Options
Reject { Assert-EcmBuildProfile $wrongLegacy $wrongStage1 } 'architecture/tiers/PRAC differs' 'legacy tier mismatch is refused before adoption'
Check (-not (Test-Path -LiteralPath (Join-Path $wrongLegacy 'profile.json'))) 'failed legacy admission creates no compatibility marker'
Write-Fixture $nvcc 'changed compiler at same version and path'
Check ((Profile).Hash -ne $a.Hash) 'compiler binary replacement changes configuration identity'
Write-Fixture $nvcc 'CUDA compiler identity'
$gmp.Hashes['lib/gmp.lib']='replacement library'
Check ((Profile).Hash -ne $a.Hash) 'GMP binary identity affects compatibility'
$gmp.Hashes['lib/gmp.lib']='library'
Assert-EcmSeparatePackageDirectory $leaf (Join-Path $fixture 'dist')
Check $true 'separate package directory accepted'
Reject { Assert-EcmSeparatePackageDirectory $leaf $leaf } 'non-nested' 'package cannot equal build directory'
Reject { Assert-EcmSeparatePackageDirectory $leaf (Join-Path $leaf 'package') } 'non-nested' 'package cannot be nested inside build'
Reject { Assert-EcmSeparatePackageDirectory $leaf (Split-Path -Parent $leaf) } 'non-nested' 'build cannot be nested inside package'
if ($AlternatePowerShell) {
    $fieldsPath=Join-Path $out 'fields.json'; Write-Fixture $fieldsPath ($a.Fields | ConvertTo-Json -Depth 4)
    $script=Join-Path $out 'alternate.ps1'; $fingerprint=Join-Path $out 'alternate_hash.txt'
    Write-Fixture $script @'
param([string]$Helper,[string]$FieldsPath,[string]$Output)
. $Helper
$fields=Get-Content -LiteralPath $FieldsPath -Raw -Encoding UTF8|ConvertFrom-Json
[IO.File]::WriteAllText($Output,(Get-EcmProfileHash $fields),[Text.UTF8Encoding]::new($false))
'@
    & $AlternatePowerShell -NoProfile -ExecutionPolicy Bypass -File $script -Helper (Join-Path $repo 'tools/build/internal/cuda_build_profile.ps1') -FieldsPath $fieldsPath -Output $fingerprint
    if ($LASTEXITCODE -ne 0) { throw 'Alternate PowerShell profile hash failed' }
    Check ([IO.File]::ReadAllText($fingerprint) -eq $a.Hash) 'profile key is stable across PowerShell runtimes'
}
[ordered]@{complete=$true;checks=$checks.Count;tests=@($checks);gpu_runs=0} |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $out 'result.json') -Encoding utf8
Write-Host "PASS: $($checks.Count) profile isolation gates"
