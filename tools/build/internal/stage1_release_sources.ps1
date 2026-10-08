# Source identity shared by the Stage1 release build and SkipBuild checks.
function Get-Stage1CudaObjects([string]$BuildDir) {
    # Compile commands, rather than a directory glob, exclude stale optional kernels.
    $root = [IO.Path]::GetFullPath($BuildDir).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
    $entries = Get-Content -LiteralPath (Join-Path $root 'compile_commands.json') -Raw | ConvertFrom-Json
    $objects = @($entries | Where-Object {
        $_.file -match '\.cu$' -and $_.command.Replace('\','/') -match 'CMakeFiles/ecm_cuda\.dir/'
    } | ForEach-Object {
        $output = $_.output
        if (-not $output) {
            $match = [regex]::Match($_.command, '-o\s+("[^"]+"|\S+)')
            if (-not $match.Success) { throw "Missing CUDA object output: $($_.file)" }
            $output = $match.Groups[1].Value.Trim('"')
        }
        $path = if ([IO.Path]::IsPathRooted($output)) { $output } else { Join-Path $_.directory $output }
        $path = [IO.Path]::GetFullPath($path)
        if (-not $path.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { throw "CUDA object outside build directory: $path" }
        $path.Substring($root.Length).Replace('\','/')
    } | Sort-Object -Unique)
    if (-not $objects.Count) { throw 'No ecm_cuda CUDA objects in compile_commands.json' }
    return $objects
}

function Get-Stage1ReleaseSources([string]$Repo) {
    $files = @(Get-Item -LiteralPath (Join-Path $Repo 'CMakeLists.txt'))
    foreach ($dir in @('src','kernels','include','cgbn/include','tools/build')) {
        $files += @(Get-ChildItem -LiteralPath (Join-Path $Repo $dir) -Recurse -File |
            Where-Object { $_.Extension -in @('.c','.cpp','.h','.hpp','.cu','.cuh','.cmake','.ps1') })
    }
    foreach ($name in @('config/ecm_options.json','config/ecm_config.generated.json')) {
        $files += Get-Item -LiteralPath (Join-Path $Repo $name)
    }
    $hashes = [ordered]@{}
    foreach ($file in ($files | Sort-Object FullName -Unique)) {
        $relative = $file.FullName.Substring($Repo.TrimEnd('\','/').Length + 1).Replace('\','/')
        $hashes[$relative] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    }
    return $hashes
}
function Assert-Stage1ReleaseSources($Expected, $Actual) {
    $names = @($Expected.PSObject.Properties.Name)
    if ($names.Count -ne $Actual.Count) { throw 'Stage1 source file set changed; rebuild before packaging' }
    foreach ($name in $names) {
        if ($Expected.$name -ne $Actual[$name]) { throw "Stage1 source changed: $name; rebuild before packaging" }
    }
}
