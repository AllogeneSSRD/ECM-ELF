#Requires -Version 5.1
# Shared release archive writer. Only explicit release payload enters the ZIP.
param(
    [Parameter(Mandatory=$true)][string]$PackageDir,
    [Parameter(Mandatory=$true)][string]$BuildDir,
    [Parameter(Mandatory=$true)][string[]]$Files,
    [Parameter(Mandatory=$true)][ValidateSet('stage1','stage2')][string]$Stage
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'release_payload.ps1')
Assert-EcmReleasePayload $Files
$PackageDir = [IO.Path]::GetFullPath($PackageDir)
$BuildDir = [IO.Path]::GetFullPath($BuildDir)
$staging = Join-Path $BuildDir ('_packages/' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $staging -Force | Out-Null
foreach ($name in $Files) {
    $base = [IO.Path]::GetFullPath($staging).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
    $target = [IO.Path]::GetFullPath((Join-Path $base $name))
    if ([IO.Path]::IsPathRooted($name) -or -not $target.StartsWith($base,[StringComparison]::OrdinalIgnoreCase)) { throw "Invalid archive payload path: $name" }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    Copy-Item -LiteralPath (Join-Path $PackageDir $name) -Destination $target
}
# Do not archive a user's existing TOML config, work queue, saves, logs or progress state.
Copy-Item -LiteralPath (Join-Path $staging 'ecm.toml.example') -Destination (Join-Path $staging 'ecm.toml')
$queueName = 'worktodo.txt'
$header = if ($Stage -eq 'stage1') { '# Add Stage1 ECM worktodo assignments here.' } else {
    '# Add ECMSTAGE2=k,b,n,c,filename[,B2-or-zero][,skip_curves][,num_curves][,"known-factors"].'
}
[IO.File]::WriteAllText((Join-Path $staging $queueName), $header + [Environment]::NewLine, (New-Object Text.UTF8Encoding $false))
$zip = $PackageDir.TrimEnd('\','/') + '.zip'
$payload = @(Get-ChildItem -LiteralPath $staging | ForEach-Object { $_.FullName })
Compress-Archive -LiteralPath $payload -DestinationPath $zip -CompressionLevel Optimal -Force
$digest = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText($zip + '.sha256', "$digest  $([IO.Path]::GetFileName($zip))" + [Environment]::NewLine)
Write-Host "Archive: $zip"
