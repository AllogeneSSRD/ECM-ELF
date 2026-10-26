#Requires -Version 5.1
param(
    [string]$BuildDir = 'build_gui_v2_test',
    [string]$VcVars = '',
    [string]$VulkanRoot = '',
    [string]$BackendRoot = '',
    [string]$GmpRoot = '',
    [switch]$BasePetOnly,
    [switch]$TestHelpers,
    [switch]$WithoutLocalPari
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if (-not [IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
$BuildDir = [IO.Path]::GetFullPath($BuildDir)
if ($GmpRoot) { $GmpRoot = [IO.Path]::GetFullPath($GmpRoot) }
$gmpIdentity = $null
if ($GmpRoot) {
    . (Join-Path $PSScriptRoot '../internal/gmp_dependency.ps1')
    $gmpIdentity = Resolve-EcmGmp $repo $GmpRoot -Release
}
if ($WithoutLocalPari -and (Test-Path -LiteralPath (Join-Path $BuildDir 'libpari.dll'))) { throw 'WithoutLocalPari requires a fresh build directory without a retained PARI runtime' }
$catalog = @('src/ecm/application/gui_v2','src/ecm/application/gui','src/ecm/application/configuration',
    'src/ecm/platform/generated','src/ecm/group_order','src/ecm/curve','src/ecm/diagnostics','include/ecm','third_party/imgui','third_party/imgui_vulkan','third_party/gui_v2_vulkan','assets/gui_v2','tools/packaging/diagnostics_runtime.py','tools/packaging/gmp_payload.py','tools/build/test/build_diagnostics.ps1')
if (-not $GmpRoot) { $catalog += 'third_party/gmp-zen3/dist' }
$catalog += @('tools/build/test/build_gui_v2.ps1','tools/build/internal/check_gui_v2_config.cmake',
    'config/gui_v2_localization.generated.json','config/gui_v2_state.generated.json')
$inputs = @($catalog | ForEach-Object { Get-ChildItem -LiteralPath (Join-Path $repo $_) -File -Recurse })
if($BasePetOnly) {
    $assetRoot=(Join-Path $repo 'assets/gui_v2').TrimEnd('\')+'\'
    $inputs=@($inputs|Where-Object { -not $_.FullName.StartsWith($assetRoot,[StringComparison]::OrdinalIgnoreCase) -or
        ($_.DirectoryName -eq $assetRoot.TrimEnd('\') -and $_.Name -in @('pet_ely.png','pet_ely.json','CLASSIC-ELY-NOTICE.md','ELYTerminal-LICENSE.txt')) })
}
if (-not $WithoutLocalPari) {
    $pari = Join-Path $repo 'third_party/pari-2.17.3'
    if (-not (Test-Path -LiteralPath (Join-Path $pari 'provenance.json'))) { throw 'Prepare local PARI with tools/build/test/prepare_gui_v2_pari.py first' }
    $provenance = Get-Content -LiteralPath (Join-Path $pari 'provenance.json') -Raw | ConvertFrom-Json
    foreach ($entry in $provenance.sha256.PSObject.Properties) {
        $file = Join-Path $pari $entry.Name
        if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) { throw "Local PARI identity mismatch: $($entry.Name)" }
        $inputs += Get-Item -LiteralPath $file
    }
    $inputs += Get-Item -LiteralPath (Join-Path $pari 'provenance.json')
}
$inputHashes = [ordered]@{}
foreach ($file in $inputs | Sort-Object FullName -Unique) {
    $key = $file.FullName.Substring($repo.Length+1).Replace('\','/')
    $inputHashes[$key] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
foreach ($legacy in @($repo, (Join-Path $repo 'build_gui'), (Join-Path $repo 'build_gui_local'))) {
    if ($BuildDir.TrimEnd('\') -eq $legacy.TrimEnd('\')) { throw 'GUI v2 requires an independent build directory' }
}
if (-not $VcVars) {
    $candidate = Get-ChildItem 'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat' |
        Where-Object FullName -NotMatch 'BuildTools' | Select-Object -First 1
    if (-not $candidate) { throw 'MSVC vcvars64.bat not found; pass -VcVars' }
    $VcVars = $candidate.FullName
}
New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
[IO.File]::WriteAllText((Join-Path $BuildDir '.gitignore'), "*`n")
$defs = @('-DCMAKE_BUILD_TYPE=Release')
if ($GmpRoot) { $defs += "-DECM_GROUP_ORDER_GMP_ROOT=`"$GmpRoot`"" }
if ($BasePetOnly) { $defs += '-DECM_GUI_V2_BASE_PET_ONLY=ON' } else { $defs += '-DECM_GUI_V2_BASE_PET_ONLY=OFF' }
if ($WithoutLocalPari) { $defs += '-DECM_GUI_V2_LOCAL_PARI=OFF' } else { $defs += '-DECM_GUI_V2_LOCAL_PARI=ON' }
if ($VulkanRoot) { $defs += "-DECM_GUI_V2_VULKAN=`"$VulkanRoot`"" }
if ($BackendRoot) { $defs += "-DECM_GUI_V2_BACKEND=`"$BackendRoot`"" }
$configure = "cmake -S `"$repo/src/ecm/application/gui_v2`" -B `"$BuildDir`" -G `"NMake Makefiles`" " + ($defs -join ' ')
function Invoke-Compiler([string]$Command, [string]$Log) {
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & cmd.exe /d /c "call `"$VcVars`" >nul 2>&1 && $Command" 2>&1 | Out-String
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $saved }
    [IO.File]::WriteAllText((Join-Path $BuildDir $Log), $output)
    if ($code) { Write-Host $output; throw "GUI v2 command failed ($code), log $Log" }
    $output -split "`r?`n" | Where-Object { $_ -match 'gui_v2:|Built target|error|warning' } | Write-Host
}
Invoke-Compiler $configure 'configure.log'
$targets = if ($TestHelpers) { 'ecm_gui_v2 ecm_gui_v2_fake_worker ecm_gui_v2_owner_probe ecm_gui_v2_state_test ecm_gui_v2_app_test ecm_gui_v2_gpu_clock_test ecm_gui_v2_group_order_test ecm_gui_v2_native_timeout_fixture' } else { 'ecm_gui_v2' }
Invoke-Compiler "cmake --build `"$BuildDir`" --target $targets" 'build.log'
foreach ($key in $inputHashes.Keys) {
    if ((Get-FileHash -LiteralPath (Join-Path $repo $key) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $inputHashes[$key]) {
        throw "GUI v2 source catalog changed during build: $key"
    }
}
$binary = Join-Path $BuildDir 'ecm_gui_v2.exe'
if ($gmpIdentity) {
    foreach ($name in $gmpIdentity.Hashes.Keys) {
        if ((Get-FileHash -LiteralPath (Join-Path $GmpRoot $name)).Hash -ne $gmpIdentity.Hashes[$name]) { throw "GUI GMP dependency changed during build: $name" }
    }
}
$sourceCommit = 'source archive without git metadata'
if (Test-Path -LiteralPath (Join-Path $repo '.git')) { $sourceCommit = & git -C $repo rev-parse HEAD }
elseif (Test-Path -LiteralPath (Join-Path $repo 'SOURCE_IDENTITY.json')) {
    $sourceCommit = (Get-Content -LiteralPath (Join-Path $repo 'SOURCE_IDENTITY.json') -Raw -Encoding UTF8 | ConvertFrom-Json).source_commit
}
[ordered]@{schema=1;exe_sha256=(Get-FileHash -LiteralPath $binary).Hash.ToLowerInvariant();
    diagnostics_host_sha256=(Get-FileHash -LiteralPath (Join-Path $BuildDir 'ecm_diagnostics_host.exe')).Hash.ToLowerInvariant();
    group_order_helper_sha256=(Get-FileHash -LiteralPath (Join-Path $BuildDir 'ecm_group_order_ui.exe')).Hash.ToLowerInvariant();
    group_order_runtime_sha256=(Get-FileHash -LiteralPath (Join-Path $BuildDir 'gmp-10.dll')).Hash.ToLowerInvariant();
    native_pari_enabled=(-not $WithoutLocalPari);
    gmp_root=$GmpRoot;gmp_hashes=$(if($gmpIdentity){$gmpIdentity.Hashes}else{$null});base_pet_only=[bool]$BasePetOnly;
    native_pari_sha256=$(if (-not $WithoutLocalPari) { (Get-FileHash -LiteralPath (Join-Path $BuildDir 'libpari.dll')).Hash.ToLowerInvariant() } else { $null });
    source_commit=$sourceCommit;source_catalog_sha256=$inputHashes;
    boundary='Conservative repository source catalog frozen before configure and checked after build; not a claim that every catalog file is linked';
    vcvars=$VcVars;cmake_cache_sha256=(Get-FileHash -LiteralPath (Join-Path $BuildDir 'CMakeCache.txt')).Hash.ToLowerInvariant()
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $BuildDir 'gui_v2_build_manifest.json') -Encoding utf8
Write-Host "GUI v2 executable: $(Join-Path $BuildDir 'ecm_gui_v2.exe')"
