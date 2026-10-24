# Shared by Stage1/Stage2 release packagers. Validate frozen host/source identity
# and copy only explicitly recorded payload; no workflow/config files enter it.
function Copy-EcmDiagnosticsPayload([string]$Repo,[string]$Build,[string]$Destination) {
    $identity = Get-Content -LiteralPath (Join-Path $Build 'diagnostics_build_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($identity.schema -ne 1 -or -not $identity.local_pari) { throw 'Release requires a complete PARI diagnostic runtime' }
    foreach ($entry in $identity.sources_sha256.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath (Join-Path $Repo $entry.Name)).Hash.ToLowerInvariant() -ne $entry.Value) { throw "Diagnostic source changed: $($entry.Name)" }
    }
    $base = [IO.Path]::GetFullPath($Destination).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
    foreach ($entry in $identity.files_sha256.PSObject.Properties) {
        $source = Join-Path $Build $entry.Name
        $target = [IO.Path]::GetFullPath((Join-Path $base $entry.Name))
        if (-not $target.StartsWith($base,[StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid diagnostic payload path' }
        if ((Get-FileHash -LiteralPath $source).Hash.ToLowerInvariant() -ne $entry.Value) { throw "Diagnostic runtime changed: $($entry.Name)" }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
        if ([IO.Path]::GetFullPath($source) -ine $target) { Copy-Item -LiteralPath $source -Destination $target -Force }
        $entry.Name
    }
}
