# Shared policy: only the INI reference is distributed as user documentation.
function Remove-EcmReleaseMetadata([string]$Directory) {
    # Remove obsolete generated metadata/docs from older packages, without recursion.
    $root = [IO.Path]::GetFullPath($Directory).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
    foreach ($file in @(Get-ChildItem -LiteralPath $root -File)) {
        if ($file.Name -like '*manifest.json' -or $file.Name -like 'DEV_*' -or
            ($file.Extension -ieq '.md' -and $file.Name -ine 'ECM_INI_REFERENCE.md')) {
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
            ([IO.Path]::GetExtension($name) -ieq '.md' -and $name -ine 'ECM_INI_REFERENCE.md')) {
            throw "Internal metadata/documentation is not a release payload: $name"
        }
    }
}
