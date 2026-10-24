# Shared source inventory, dependency closure and object identity. PowerShell 5.1.
# Dependency scanning is conservative across preprocessor branches. Compiler
# depfiles supplement it with macro/system includes; they are never substituted
# for scanning current local includes (which detects newly introduced headers).
function Get-EcmCudaInventory([string]$Repo, [string]$Output) {
    $temporary = -not $Output
    if ($temporary) { $Output = [IO.Path]::GetTempFileName() }
    try {
        & cmake "-DECM_CUDA_SOURCES_OUTPUT=$Output" -P (Join-Path $Repo 'tools/build/internal/cuda_sources.cmake')
        if ($LASTEXITCODE -ne 0) { throw 'CUDA source inventory export failed' }
        return Get-Content -LiteralPath $Output -Raw | ConvertFrom-Json
    } finally { if ($temporary) { Remove-Item -LiteralPath $Output -Force } }
}

function Get-EcmSourceKey([string]$Repo, [string]$Path) {
    $prefix = [IO.Path]::GetFullPath($Repo).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
    $absolute = [IO.Path]::GetFullPath($Path)
    if ($absolute.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        return $absolute.Substring($prefix.Length).Replace('\','/')
    }
    return $absolute.Replace('\','/')
}

function Resolve-EcmSourcePath([string]$Repo, [string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $Repo $Path))
}

function Get-EcmSourceClosure([string]$Repo, [string[]]$Sources, [string[]]$IncludeDirs = @(), [switch]$Production) {
    $dirs = @($IncludeDirs | ForEach-Object { Resolve-EcmSourcePath $Repo $_ })
    $pending = New-Object 'System.Collections.Generic.Stack[string]'
    foreach ($source in $Sources) { $pending.Push((Resolve-EcmSourcePath $Repo $source)) }
    $seen = @{}
    while ($pending.Count) {
        $path = $pending.Pop()
        if ($seen.ContainsKey($path)) { continue }
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Build dependency missing: $path" }
        $key = Get-EcmSourceKey $Repo $path
        if ($Production -and $key -match '^(tools/(bench|experiments|test)/|\.refactor/)') {
            throw "Production CUDA source depends on experiment: $key"
        }
        $seen[$path] = $key
        if ([IO.Path]::GetExtension($path) -notin @('.cu','.cuh','.cpp','.c','.h','.hpp','.inl','.inc','.def')) { continue }
        $content = [IO.File]::ReadAllText($path)
        $content = [regex]::Replace($content, '(?s)/\*.*?\*/', '')
        foreach ($match in [regex]::Matches($content, '(?m)^\s*#\s*include\s*(["<])([^">\r\n]+)[">]')) {
            $name = $match.Groups[2].Value
            $search = @()
            if ($match.Groups[1].Value -eq '"') { $search += Split-Path -Parent $path }
            $search += $dirs
            $resolved = $null
            foreach ($dir in $search) {
                $candidate = Resolve-EcmSourcePath $dir $name
                if (Test-Path -LiteralPath $candidate -PathType Leaf) { $resolved = $candidate; break }
            }
            if ($resolved) { $pending.Push($resolved) }
            elseif ($match.Groups[1].Value -eq '"' -and -not [IO.Path]::IsPathRooted($key)) {
                throw "Unresolved local include '$name' in $key"
            }
            # SDK/standard angle headers are added from compiler depfiles. An
            # inactive platform include is allowed to be absent on this host.
            # The same applies to quoted includes *inside external SDK headers*
            # (e.g. CUDA runtime's NVRTC-only "target"). Active SDK dependencies
            # must still be discovered by the successful compiler and hashed.
        }
    }
    return @($seen.Values | Sort-Object -Unique)
}

function Get-EcmSourceHashes([string]$Repo, [string[]]$Paths) {
    $hashes = [ordered]@{}
    $orderedPaths = [string[]]@($Paths | Select-Object -Unique)
    [Array]::Sort($orderedPaths, [StringComparer]::Ordinal)
    foreach ($path in $orderedPaths) {
        $hashes[$path] = (Get-FileHash -LiteralPath (Resolve-EcmSourcePath $Repo $path) -Algorithm SHA256).Hash
    }
    return $hashes
}

function Assert-EcmHashes($Expected, $Actual, [string]$Message = 'Build dependency changed') {
    $expectedNames = if ($Expected -is [System.Collections.IDictionary]) { @($Expected.Keys) } else { @($Expected.PSObject.Properties.Name) }
    $actualNames = @($Actual.Keys)
    foreach ($name in $actualNames) {
        if ($Expected.$name -ne $Actual[$name]) { throw "${Message}: $name" }
    }
    if ($expectedNames.Count -ne $actualNames.Count) { throw "${Message}: dependency set" }
}

function Get-EcmCommandInputs($Entry) {
    $pending = New-Object 'System.Collections.Generic.Queue[string]'
    $pending.Enqueue([string]$Entry.command)
    $responses = @{}
    $texts = @()
    while ($pending.Count) {
        $text = $pending.Dequeue(); $texts += $text
        foreach ($m in [regex]::Matches($text, '(?:--options-file(?:=|\s+)|-optf\s+|@)(?:"([^"]+)"|([^\s"]+))')) {
            $name = if ($m.Groups[1].Success) { $m.Groups[1].Value } else { $m.Groups[2].Value }
            foreach ($part in $name.Split(',')) {
                $path = Resolve-EcmSourcePath $Entry.directory $part
                if (-not $responses.ContainsKey($path)) {
                    $responses[$path] = $true
                    $pending.Enqueue([IO.File]::ReadAllText($path))
                }
            }
        }
    }
    $dirs = @()
    foreach ($text in $texts) {
        foreach ($m in [regex]::Matches($text, '(?:^|\s)(?:-I|/I|--include-path(?:=|\s+))\s*(?:"([^"]+)"|([^\s"]+))')) {
            $name = if ($m.Groups[1].Success) { $m.Groups[1].Value } else { $m.Groups[2].Value }
            $dirs += Resolve-EcmSourcePath $Entry.directory $name
        }
    }
    return [pscustomobject]@{ IncludeDirs = @($dirs | Select-Object -Unique); Responses = @($responses.Keys | Sort-Object) }
}

function Get-EcmObjectPath($Entry, [string]$BuildDir = '') {
    $output = $Entry.output
    if (-not $output) {
        $m = [regex]::Match($Entry.command, '(?:-o\s+|/Fo)(?:"([^"]+)"|([^\s"]+))')
        if (-not $m.Success) { throw "Missing object output: $($Entry.file)" }
        $output = if ($m.Groups[1].Success) { $m.Groups[1].Value } else { $m.Groups[2].Value }
    }
    $path = Resolve-EcmSourcePath $Entry.directory $output
    if ($BuildDir) {
        $prefix = [IO.Path]::GetFullPath($BuildDir).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
        if (-not $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Object outside build directory: $path" }
    }
    return $path
}

function Get-EcmTargetCompileEntries([string]$BuildDir, [string]$Target) {
    $entries = Get-Content -LiteralPath (Join-Path $BuildDir 'compile_commands.json') -Raw | ConvertFrom-Json
    $graphPath = Join-Path $BuildDir 'cuda_target_graph.json'
    $targets = @($Target)
    if (Test-Path -LiteralPath $graphPath) {
        $graph = Get-Content -LiteralPath $graphPath -Raw | ConvertFrom-Json
        if ($graph.targets.$Target) { $targets = @($graph.targets.$Target) }
    }
    $pattern = 'CMakeFiles/(' + (($targets | ForEach-Object { [regex]::Escape($_) }) -join '|') + ')\.dir/'
    return @($entries | Where-Object { $_.command.Replace('\','/') -match $pattern })
}

function Remove-EcmCudaObjects([string]$BuildDir, [string]$Target = 'ecm_cuda') {
    foreach ($entry in @(Get-EcmTargetCompileEntries $BuildDir $Target | Where-Object { $_.file -match '\.cu$' })) {
        $path = Get-EcmObjectPath $entry $BuildDir
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
    }
}

function Get-EcmDepfilePaths([string]$Path, [string]$Directory) {
    if (-not (Test-Path -LiteralPath $Path)) { return @() }
    $text = [IO.File]::ReadAllText($Path) -replace '\\\r?\n', ' '
    # The target may contain a drive colon; the rule separator is colon+space.
    $separator = [regex]::Match($text, ':\s')
    if (-not $separator.Success) { throw "Invalid compiler dependency file: $Path" }
    $text = $text.Substring($separator.Index + $separator.Length)
    return @([regex]::Matches($text, '(?:\\[ \t#:]|[^\s])+') | ForEach-Object {
        $name = $_.Value -replace '\\([ \t#:])', '$1'
        Resolve-EcmSourcePath $Directory $name
    } | Sort-Object -Unique)
}

function Get-EcmCompileState([string]$Repo, $Entry, [string]$Setup = '', [string[]]$IncludeDirs = @(), [switch]$Production) {
    $inputs = Get-EcmCommandInputs $Entry
    $object = Get-EcmObjectPath $Entry
    $paths = @(Get-EcmSourceClosure $Repo @($Entry.file) (@($inputs.IncludeDirs) + $IncludeDirs) -Production:$Production)
    $paths += @($inputs.Responses | ForEach-Object { Get-EcmSourceKey $Repo $_ })
    $compilerPaths = if ($Entry.command -match '(?:^|\s)/(?:sourceDependencies\s|showIncludes(?:\s|$))') {
        Get-EcmMsvcDependencyPaths "$object.ecm-msvc.json" $Entry
    } else { Get-EcmDepfilePaths "$object.ecm.d" $Entry.directory }
    $paths += @($compilerPaths |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        ForEach-Object { Get-EcmSourceKey $Repo $_ })
    # A removed old compiler dependency invalidates the saved fingerprint, but
    # must not prevent recompiling a source which no longer includes it.
    if ($Production) {
        foreach ($path in $paths) {
            if ($path -match '^(tools/(bench|experiments|test)/|\.refactor/)') { throw "Production CUDA source depends on experiment: $path" }
        }
    }
    $hashes = Get-EcmSourceHashes $Repo $paths
    # Locale collation and JSON escaping differ between Windows PowerShell and
    # PowerShell 7. Length-framed UTF-8 fields in ordinal path order have one
    # representation on both runtimes, including quotes, Unicode and newlines.
    $parts = @('ECM-CUDA-STATE-2', [string]$Entry.directory, [string]$Entry.command, $Setup)
    foreach ($name in $hashes.Keys) { $parts += $name; $parts += $hashes[$name] }
    $buffer = New-Object Text.StringBuilder
    foreach ($part in $parts) {
        [void]$buffer.Append($part.Length.ToString([Globalization.CultureInfo]::InvariantCulture)).Append(':').Append($part)
    }
    $text = $buffer.ToString()
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $fingerprint = [BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($text))).Replace('-','') }
    finally { $sha.Dispose() }
    # Source root is live validation context, not part of the saved object record
    # or fingerprint. It resolves inputs retired from the new compiler depfile.
    return [pscustomobject]@{ schema = 2; fingerprint = $fingerprint; source_hashes = $hashes; object = $object;
        source_root = [IO.Path]::GetFullPath($Repo) }
}

function Get-EcmMsvcDependencyPaths([string]$Path, $Entry) {
    if (-not (Test-Path -LiteralPath $Path)) { return @() }
    $dependencies = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($dependencies.Version -notin @('1.1','1.2') -or -not $dependencies.Data.Source -or
        'Includes' -notin @($dependencies.Data.PSObject.Properties.Name) -or
        [IO.Path]::GetFullPath($dependencies.Data.Source) -ine [IO.Path]::GetFullPath($Entry.file)) {
        throw "Invalid MSVC compiler dependency file: $Path"
    }
    if (@($dependencies.Data.ImportedModules | Where-Object { $_ }).Count -or
        @($dependencies.Data.ImportedHeaderUnits | Where-Object { $_ }).Count) {
        throw "MSVC module dependencies require an explicit build contract: $Path"
    }
    return @($dependencies.Data.Source) + @($dependencies.Data.Includes)
}

function Get-EcmCompileInvalidationReason($State) {
    $record = "$($State.object).ecm-state.json"
    if (-not (Test-Path -LiteralPath $State.object)) { return 'object missing' }
    if (-not (Test-Path -LiteralPath $record)) { return 'dependency record missing' }
    try {
        $saved = Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
        if ($saved.schema -ne 2) { return 'dependency record schema changed' }
        try { Assert-EcmHashes $saved.source_hashes $State.source_hashes }
        catch { return "source or dependency changed: $($_.Exception.Message)" }
        if ($saved.fingerprint -ne $State.fingerprint) { return 'compile command or setup changed' }
        if ($saved.object_sha256 -ne (Get-FileHash -LiteralPath $State.object -Algorithm SHA256).Hash) { return 'object content changed' }
        return ''
    } catch { return "dependency record invalid: $($_.Exception.Message)" }
}

function Test-EcmCompileState($State) {
    return (Get-EcmCompileInvalidationReason $State) -eq ''
}

function Save-EcmCompileState($State) {
    $record = [ordered]@{ schema = 2; fingerprint = $State.fingerprint; source_hashes = $State.source_hashes;
        object_sha256 = (Get-FileHash -LiteralPath $State.object -Algorithm SHA256).Hash }
    [IO.File]::WriteAllText("$($State.object).ecm-state.json", ($record | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
}

function Assert-EcmCompileInputs($Before, $After) {
    # A successful compiler may discover additional SDK/macro includes. Every
    # dependency known before compilation must still have the same content,
    # including a retired include absent from the new dependency set.
    foreach ($name in $Before.source_hashes.Keys) {
        if ($After.source_hashes.Contains($name)) {
            $actual = $After.source_hashes[$name]
        } else {
            if (-not $Before.source_root) { throw "Cannot validate retired build dependency without source root: $name" }
            $path = Resolve-EcmSourcePath $Before.source_root $name
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Source changed during build (missing): $name" }
            $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        }
        if ($Before.source_hashes[$name] -ne $actual) { throw "Source changed during build: $name" }
    }
}

function Get-EcmStage2SourceSet([string]$Repo, $Inventory, [string]$Engine, [string[]]$IncludeDirs = @(), [string]$Modules = '') {
    $group = if ($Engine -eq 'production') { 'STAGE2_PRODUCTION' } else { 'STAGE2_DEVELOPMENT' }
    $engineSources = @($Inventory.groups.$group)
    if ($Modules) {
        $moduleInventory = Get-Content -LiteralPath (Join-Path $Modules 'modules.json') -Raw | ConvertFrom-Json
        foreach ($inputFile in $moduleInventory.inputs_sha256.PSObject.Properties) {
            if ((Get-FileHash -LiteralPath (Join-Path $Repo $inputFile.Name)).Hash -ine $inputFile.Value) { throw "CUDA module generator input changed: $($inputFile.Name)" }
        }
        $engineSources = @($engineSources | Where-Object { $_ -notmatch '/(arithmetic_kernels|transform_kernels)\.cu$' }) + @($moduleInventory.modules.source)
    }
    $cudaSources = @($engineSources | Where-Object { $_ -match '\.cu$' })
    $sources = @($engineSources) + @($Inventory.groups.STAGE2_HOST)
    $cudaDeps = @(Get-EcmSourceClosure $Repo $cudaSources $IncludeDirs -Production:($Engine -eq 'production'))
    $deps = @(Get-EcmSourceClosure $Repo $sources $IncludeDirs -Production:($Engine -eq 'production')) +
        @($Inventory.build_inputs) + @('tools/build/build_stage2_local.ps1')
    if ($Modules) { $deps += @($moduleInventory.inputs_sha256.PSObject.Properties.Name) + @((Join-Path $Modules 'modules.json')) }
    return [pscustomobject]@{ Sources = $sources; CudaSources = $cudaSources;
        Dependencies = @($deps | Sort-Object -Unique); CudaDependencies = $cudaDeps }
}

function Get-EcmObjectNames([string[]]$Sources) {
    $names = [ordered]@{}
    foreach ($source in $Sources) {
        $stem = [IO.Path]::GetFileNameWithoutExtension($source)
        if (@($Sources | Where-Object { [IO.Path]::GetFileNameWithoutExtension($_) -ieq $stem }).Count -gt 1) {
            $stem = $source.Replace('\','/') -replace '[^A-Za-z0-9_.-]', '_'
        }
        if ($names.Values -icontains $stem) { throw "Ambiguous object name: $source" }
        $names[$source] = $stem
    }
    return $names
}

function Assert-EcmObjectHashes([string]$BuildDir, $Objects) {
    foreach ($entry in $Objects.PSObject.Properties) {
        $path = Resolve-EcmSourcePath $BuildDir $entry.Name
        $prefix = [IO.Path]::GetFullPath($BuildDir).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
        if (-not $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Object outside build directory: $path" }
        if ($entry.Value -ne (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash) { throw "Build object changed: $($entry.Name)" }
    }
}

function Assert-EcmBuildUnits([string]$Repo, [string]$BuildDir, $Manifest,
        [string[]]$Sources, [string[]]$IncludeDirs = @(), [switch]$Production) {
    if ($Manifest.identity_schema -ne 1 -or -not $Manifest.compile_units) {
        throw 'Build identity is missing dependency records; rebuild old manifests'
    }
    if (@($Manifest.compile_units.PSObject.Properties).Count -ne $Sources.Count) { throw 'Build translation-unit set changed' }
    foreach ($source in $Sources) {
        $unit = $Manifest.compile_units.$source
        if (-not $unit -or $unit.entry.file -ine (Resolve-EcmSourcePath $Repo $source)) { throw "Build translation unit changed: $source" }
        $object = Get-EcmObjectPath $unit.entry $BuildDir
        $state = Get-EcmCompileState $Repo $unit.entry $Manifest.compile_setup $IncludeDirs -Production:$Production
        if ($unit.fingerprint -ne $state.fingerprint) { throw "Build compile dependency changed: $source" }
        if ($unit.object_sha256 -ne (Get-FileHash -LiteralPath $object -Algorithm SHA256).Hash) { throw "Build object changed: $source" }
    }
}
