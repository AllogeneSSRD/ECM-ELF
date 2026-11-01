# Local build output policy. No compiler/profile state is modified here.
function Initialize-EcmBuildDirectory([Parameter(Mandatory=$true)][string]$Path) {
    $full=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
    $full=[IO.Path]::GetFullPath($full).TrimEnd('\','/')
    $repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..')).TrimEnd('\','/')
    if($full -eq [IO.Path]::GetPathRoot($full).TrimEnd('\','/') -or $full -ieq $repo -or
       $repo.StartsWith($full+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){
        throw 'Repository and filesystem roots cannot be ignored build directories'
    }
    foreach($source in @('.git','.agents','.codex','src','include','config','docs','kernels','tools')){
        $protected=Join-Path $repo $source
        if($full -ieq $protected -or $full.StartsWith($protected+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){
            throw "Project source directory cannot be ignored: $full"
        }
    }
    $directories=New-Object 'Collections.Generic.List[string]'
    $directories.Add($full)
    # Nested test/profile leaves also protect their enclosing build_* container.
    $parent=[IO.Path]::GetDirectoryName($full)
    while($parent -and $parent.StartsWith($repo+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){
        if([IO.Path]::GetFileName($parent) -match '^build(?:_|$)'){$directories.Add($parent)}
        $parent=[IO.Path]::GetDirectoryName($parent)
    }
    $directories.Reverse()
    foreach($directory in $directories){
        [void][IO.Directory]::CreateDirectory($directory)
        $ignore=Join-Path $directory '.gitignore'
        if([IO.File]::Exists($ignore) -and [IO.File]::ReadAllText($ignore) -in @('*',"*`n","*`r`n")){continue}
        [IO.File]::WriteAllText($ignore,"*`n",[Text.UTF8Encoding]::new($false))
    }
}

# Rename build output within one workspace. Never merge directories, follow a
# junction, or move tracked source roots while reorganizing caches.
function Move-EcmBuildDirectory([string]$WorkspaceRoot, [string]$Source, [string]$Destination) {
    $root = [IO.Path]::GetFullPath($WorkspaceRoot).TrimEnd('\','/')
    $from = [IO.Path]::GetFullPath($Source).TrimEnd('\','/')
    $to = [IO.Path]::GetFullPath($Destination).TrimEnd('\','/')
    $prefix = $root + [IO.Path]::DirectorySeparatorChar
    foreach ($path in @($from, $to)) {
        if (-not $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or
            $path.Substring($prefix.Length).Split([char[]]'\/')[0] -notmatch '^build(?:_|$)') {
            throw "Build move must stay inside workspace build directories: $path"
        }
        $parent = $path
        while ($parent -and $parent.Length -ge $root.Length) {
            if ((Test-Path -LiteralPath $parent) -and
                ((Get-Item -LiteralPath $parent -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                throw "Build move cannot use a junction or symbolic link: $parent"
            }
            $parent = [IO.Path]::GetDirectoryName($parent)
        }
    }
    if ($from -ieq $to -or $to.StartsWith($from+'\', [StringComparison]::OrdinalIgnoreCase) -or
        $from.StartsWith($to+'\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Build move requires separate non-nested directories'
    }
    if (-not (Test-Path -LiteralPath $from -PathType Container)) { throw "Build source missing: $from" }
    if (Test-Path -LiteralPath $to) { throw "Build destination already exists: $to" }
    $parent = [IO.Path]::GetDirectoryName($to)
    if ($parent -ine $root) { Initialize-EcmBuildDirectory $parent }
    Initialize-EcmBuildDirectory $from
    Move-Item -LiteralPath $from -Destination $to -ErrorAction Stop
}
