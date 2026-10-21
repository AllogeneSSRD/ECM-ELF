# Configuration compatibility is distinct from source/TU and executable identity.
# Values are flat strings so PS 5.1 and PS 7 produce the same framed digest.
function Get-EcmProfileHash($Fields) {
    $names = if ($Fields -is [Collections.IDictionary]) { [string[]]@($Fields.Keys) }
        else { [string[]]@($Fields.PSObject.Properties.Name) }
    [Array]::Sort($names, [StringComparer]::Ordinal)
    $text = New-Object Text.StringBuilder
    foreach ($name in $names) {
        foreach ($value in @($name, [string]$Fields.$name)) {
            [void]$text.Append($value.Length.ToString([Globalization.CultureInfo]::InvariantCulture)).Append(':').Append($value)
        }
    }
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($text.ToString()))).Replace('-','').ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function New-EcmCudaProfile([string]$Repo,
        [ValidateSet('local','dev','release')][string]$Role,
        [ValidateSet('stage1','stage2')][string]$Stage,
        [ValidateSet('production','development')][string]$Engine,
        [string]$Archs, $Cuda, $Vc, $GmpDependency, $Options) {
    if ($Role -eq 'release' -and $Engine -ne 'production') { throw 'Release profiles require the production engine' }
    $architectures = $Archs -replace 'sm_', ''
    $fields = [ordered]@{
        role = $Role; stage = $Stage; engine = $Engine; architectures = $architectures
        source_root = [IO.Path]::GetFullPath($Repo)
        cuda_root = $Cuda.Root; toolkit = $Cuda.VersionText.Replace("`r`n", "`n")
        nvcc_sha256 = (Get-FileHash -LiteralPath $Cuda.Nvcc).Hash
        host_compiler = $Vc.Compiler; host_version = $Vc.Version
        host_compiler_sha256 = (Get-FileHash -LiteralPath $Vc.Compiler).Hash
        gmp_root = $GmpDependency.Root
    }
    if ($Stage -eq 'stage1') {
        $cmake = (Get-Command cmake -ErrorAction Stop).Source
        $versionText = (& $cmake --version | Out-String).Trim()
        $version = [regex]::Match($versionText, '^cmake version ([^\s]+)')
        if ($LASTEXITCODE -ne 0 -or -not $version.Success) { throw 'Cannot identify CMake for the Stage1 profile' }
        $fields['cmake_version'] = $version.Groups[1].Value
        $fields['cmake_path'] = $cmake
        $fields['cmake_sha256'] = (Get-FileHash -LiteralPath $cmake).Hash
    }
    foreach ($name in $GmpDependency.Hashes.Keys) { $fields["gmp/$name"] = $GmpDependency.Hashes[$name] }
    foreach ($name in $Options.Keys) { $fields["option/$name"] = [string]$Options[$name] }
    $hash = Get-EcmProfileHash $fields
    $archLabel = 'sm' + $architectures.Replace(',', '-sm')
    $relative = if ($Role -eq 'release') {
        $version = [regex]::Match($Cuda.VersionText, 'release\s+(\d+)\.(\d+)')
        if (-not $version.Success) { throw 'Cannot identify the CUDA toolkit group' }
        "$Stage/cuda$($version.Groups[1].Value)_$($version.Groups[2].Value)/$archLabel/$($hash.Substring(0,16))"
    } elseif ($Stage -eq 'stage2') {
        $label = if ($Engine -eq 'development') { 'experimental' } else { 'production' }
        "$Stage/$label/$archLabel/$($hash.Substring(0,16))"
    } else { "$Stage/$archLabel/$($hash.Substring(0,16))" }
    return [pscustomobject]@{ Hash = $hash; Key = $hash.Substring(0,16); Fields = $fields
        Role = $Role; Stage = $Stage; Engine = $Engine; Relative = $relative }
}

function New-EcmStage2Profile([string]$Repo, [string]$Role, [string]$Engine, [string]$Arch,
        [string]$GlBackend, [int]$OuterUnrollU, [int]$AddSubMask, [int]$SplitCompile,
        $Cuda, $Vc, $GmpDependency) {
    if ($Engine -eq 'production' -and ($GlBackend -ne 'ptx' -or $OuterUnrollU -ne 0 -or $AddSubMask -ne 1)) {
        throw 'Production requires -GlBackend ptx -OuterUnrollU 0 -AddSubMask 1; use -Engine development for comparisons'
    }
    $mode = @{runtime=-1;fold=0;short=1;ptx=3}[$GlBackend]
    $gmpRoot = $GmpDependency.Root
    $compile = "`"$($Cuda.Nvcc)`" -ccbin `"$($Vc.Compiler)`" -std=c++17 -O3 -arch=$Arch -DNTT_GL_FIXED_MODE=$mode -DNTT_OUTER_UNROLL_U=$OuterUnrollU -DNTT_GL_ADD_SUB_MASK=$AddSubMask " +
        "--split-compile=$SplitCompile -I `"$gmpRoot/include`" -I `"$Repo/include`" -Xcompiler /utf-8 -Xcompiler /wd4819"
    $link = "$($Vc.Setup) && `"$($Cuda.Nvcc)`" -ccbin `"$($Vc.Compiler)`" -std=c++17 -O3 -arch=$Arch"
    $options = [ordered]@{ generator = 'direct-nvcc'; configuration = 'optimized'; host_driver = 'nvcc'
        gl_backend = $GlBackend; gl_fixed_mode = $mode; outer_unroll_u = $OuterUnrollU
        add_sub_mask = $AddSubMask; split_compile = $SplitCompile; compile = $compile; link = $link }
    $buildProfile = New-EcmCudaProfile $Repo $Role stage2 $Engine $Arch $Cuda $Vc $GmpDependency $options
    $buildProfile | Add-Member -NotePropertyName CompilePrefix -NotePropertyValue $compile
    $buildProfile | Add-Member -NotePropertyName LinkPrefix -NotePropertyValue $link
    return $buildProfile
}

function Resolve-EcmBuildDirectory([string]$Repo, $buildProfile, [string]$Leaf = '', [string]$Root = '') {
    if ($Leaf -and $Root) { throw 'Specify a final build directory or a build root, not both' }
    if (-not $Leaf) {
        if (-not $Root) { $Root = "build_cuda_$($buildProfile.Role)" }
        $Leaf = Join-Path $Root $buildProfile.Relative
    }
    if (-not [IO.Path]::IsPathRooted($Leaf)) { $Leaf = Join-Path $Repo $Leaf }
    $leafPath = [IO.Path]::GetFullPath($Leaf).TrimEnd('\','/')
    if ($leafPath -ieq [IO.Path]::GetFullPath($Repo).TrimEnd('\','/') -or $leafPath -eq [IO.Path]::GetPathRoot($leafPath).TrimEnd('\','/')) {
        throw 'Repository and filesystem roots cannot be build directories'
    }
    foreach ($part in $leafPath.Split([char[]]'\/')) {
        if ($part -match '^build_cuda_(local|dev|release)$' -and $Matches[1] -ne $buildProfile.Role) {
            throw "Build directory belongs to another role: $part; requested $($buildProfile.Role)"
        }
    }
    return $leafPath
}

function Assert-EcmLegacyProfile([string]$Leaf, $buildProfile) {
    $f = $buildProfile.Fields
    $release = $buildProfile.Role -eq 'release'
    $name = if ($buildProfile.Stage -eq 'stage2') { 'build_manifest.json' }
        elseif ($release) { 'stage1_build_manifest.json' } else { 'stage1_local_manifest.json' }
    $manifestPath = Join-Path $Leaf $name
    if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'Unmarked build has no matching formal manifest; choose a fresh directory' }
    $m = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($m.role -and $m.role -ne $buildProfile.Role) { throw 'Legacy build belongs to another role' }
    if ($m.identity_schema -ne 1 -or $m.cuda_root -ine $f.cuda_root -or
        $m.toolkit.Replace("`r`n", "`n") -cne $f.toolkit -or $m.host_compiler -ine $f.host_compiler -or $m.host_version -ne $f.host_version) {
        throw 'Legacy build toolchain identity differs; choose a fresh directory'
    }
    if ($buildProfile.Stage -eq 'stage2') {
        foreach ($name in @('engine','architecture','gl_backend','gl_fixed_mode','outer_unroll_u','add_sub_mask','split_compile')) {
            $expected = if ($name -eq 'engine') { $f.engine } elseif ($name -eq 'architecture') { 'sm_' + $f.architectures } else { $f."option/$name" }
            if ([string]$m.$name -cne [string]$expected) { throw "Legacy Stage2 profile differs: $name" }
        }
        $oldRole = if ($Leaf -match '(^|[\\/])(release[^\\/]*|build_cuda_release)([\\/]|$)') { 'release' }
            elseif ($Leaf -match '(^|[\\/])(development_stage2|build_cuda_dev)([\\/]|$)') { 'dev' } else { 'local' }
        if ($oldRole -ne $buildProfile.Role) { throw 'Legacy build belongs to another role' }
        foreach ($name in @('include/gmp.h','lib/gmp.lib','bin/gmp-10.dll')) {
            if ($m.gmp_hashes.$name -ne $f."gmp/$name") { throw "Legacy GMP identity differs: $name" }
        }
    } else {
        if (-not $release -and -not $m.role) {
            $oldRole = if ($Leaf -match '(^|[\\/])build_cuda_dev([\\/]|$)') { 'dev' } else { 'local' }
            if ($oldRole -ne $buildProfile.Role) { throw 'Legacy build belongs to another role' }
        }
        $arch = if ($release) { $m.architectures } else { $m.architecture }
        if ($arch -ne $f.architectures -or $m.tiers -ne $f.'option/tiers' -or
            [bool]$m.enable_prac -ne ($f.'option/enable_prac' -eq '1')) { throw 'Legacy Stage1 architecture/tiers/PRAC differs' }
        if ($f.'option/extra') { throw 'Legacy Extra options have no recorded profile; choose a fresh directory' }
        foreach ($entry in @(@('header','include/gmp.h'),@('library','lib/gmp.lib'),@('runtime','bin/gmp-10.dll'))) {
            if ($m.dependency_hashes.($entry[0]) -ne $f.('gmp/' + $entry[1])) { throw "Legacy GMP identity differs: $($entry[1])" }
        }
        # Every requested CMake definition must already describe this cache.
        $cache = Get-Content -LiteralPath (Join-Path $Leaf 'CMakeCache.txt') -Raw
        $cacheVersion = @('MAJOR','MINOR','PATCH') | ForEach-Object {
            $found = [regex]::Match($cache, "(?m)^CMAKE_CACHE_$($_)_VERSION:INTERNAL=([^\r\n]+)")
            $found.Groups[1].Value
        }
        if (($cacheVersion -join '.') -ne $f.cmake_version) { throw 'Legacy CMake version differs; choose a fresh directory' }
        foreach ($entry in [regex]::Matches($f.'option/configure', '-D([A-Za-z0-9_]+)(?::[A-Za-z]+)?=(?:"([^"]*)"|([^\s"]*))')) {
            $key = $entry.Groups[1].Value
            $expected = if ($entry.Groups[2].Success) { $entry.Groups[2].Value } else { $entry.Groups[3].Value }
            $actual = [regex]::Match($cache, '(?m)^' + [regex]::Escape($key) + ':[^=]+=([^\r\n]*)')
            if (-not $actual.Success -or $actual.Groups[1].Value.Replace('\','/') -ine $expected.Replace('\','/')) {
                throw "Legacy CMake profile differs: $key"
            }
        }
    }
}

function Assert-EcmBuildProfile([string]$Leaf, $buildProfile, [switch]$ReadOnly) {
    $path = Join-Path $Leaf 'profile.json'
    if (Test-Path -LiteralPath $path) {
        $marker = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
        if ($marker.schema -ne 1 -or $marker.profile_sha256 -cne $buildProfile.Hash -or
            (Get-EcmProfileHash $marker.fields) -cne $buildProfile.Hash) {
            throw 'Build profile/role/engine differs; choose its matching leaf or a fresh directory'
        }
        return
    }
    if ($ReadOnly) { throw 'A matching profile marker is required for SkipBuild; build/verify this leaf first' }
    if ((Test-Path -LiteralPath $Leaf) -and @(Get-ChildItem -LiteralPath $Leaf -Force).Count) {
        Assert-EcmLegacyProfile $Leaf $buildProfile
    }
    New-Item -ItemType Directory -Path $Leaf -Force | Out-Null
    $marker = [ordered]@{ schema = 1; profile_sha256 = $buildProfile.Hash; fields = $buildProfile.Fields }
    [IO.File]::WriteAllText($path, ($marker | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
}

function Assert-EcmSeparatePackageDirectory([string]$Build, [string]$Package) {
    $buildPath = [IO.Path]::GetFullPath($Build).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
    $packagePath = [IO.Path]::GetFullPath($Package).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
    if ($buildPath.StartsWith($packagePath, [StringComparison]::OrdinalIgnoreCase) -or
        $packagePath.StartsWith($buildPath, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Build and package directories must be separate, non-nested directories'
    }
}
