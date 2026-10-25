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
