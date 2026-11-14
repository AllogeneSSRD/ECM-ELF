# Shared policy: only the TOML references are distributed as user documentation.
$script:EcmReleaseConfigReferences = @('ECM_TOML_REFERENCE.en.md','ECM_TOML_REFERENCE.zh-CN.md')

function Copy-EcmReleaseConfigReferences([string]$Repo, [string]$Directory) {
    foreach ($name in $script:EcmReleaseConfigReferences) {
        Copy-Item -LiteralPath (Join-Path $Repo "docs/$name") -Destination $Directory -Force
        $name
    }
}

function Remove-EcmReleaseMetadata([string]$Directory) {
    # Remove obsolete generated metadata/docs from older packages, without recursion.
    $root = [IO.Path]::GetFullPath($Directory).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
    foreach ($file in @(Get-ChildItem -LiteralPath $root -File)) {
        if ($file.Name -like '*manifest.json' -or $file.Name -like 'DEV_*' -or
            ($file.Extension -ieq '.md' -and $file.Name -notin $script:EcmReleaseConfigReferences)) {
            if (-not $file.FullName.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Release cleanup outside package directory: $($file.FullName)"
            }
            Remove-Item -LiteralPath $file.FullName -Force
        }
    }
}
function Assert-EcmReleasePayload([string[]]$Files) {
    foreach ($name in $Files) {
        if ($name -like '*manifest.json' -or $name -like 'DEV_*' -or
            ([IO.Path]::GetExtension($name) -ieq '.md' -and $name -notin $script:EcmReleaseConfigReferences)) {
            throw "Internal metadata/documentation is not a release payload: $name"
        }
    }
}
