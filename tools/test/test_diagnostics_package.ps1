#Requires -Version 5.1
param([Parameter(Mandatory=$true)][string]$Build,[Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Build=[IO.Path]::GetFullPath($Build);$Output=[IO.Path]::GetFullPath($Output)
if(Test-Path -LiteralPath $Output){throw 'Use a fresh output directory'}
. (Join-Path $repo 'tools/build/internal/diagnostics_payload.ps1')
$package=Join-Path $Output 'diagnostic-candidate'
$files=@(Copy-EcmDiagnosticsPayload $repo $Build $package)
Copy-Item -LiteralPath (Join-Path $repo 'config/ecm.toml.example') -Destination $package
$files+='ecm.toml.example'
[IO.File]::WriteAllText((Join-Path $package 'ecm.toml'),'private-existing-config-sentinel')
[IO.File]::WriteAllText((Join-Path $package 'worktodo.txt'),'private-existing-queue-sentinel')
& (Join-Path $repo 'tools/build/internal/package_archive.ps1') -PackageDir $package -BuildDir $Output -Stage stage2 -Files $files
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip=[IO.Compression.ZipFile]::OpenRead($package+'.zip')
try{
    $names=@($zip.Entries|ForEach-Object{$_.FullName.Replace('\','/')})
    foreach($file in $files+@('ecm.toml','worktodo.txt')){if($file -notin $names){throw "ZIP missing $file"}}
    foreach($file in @('ecm.toml','worktodo.txt')){
        $reader=New-Object IO.StreamReader($zip.GetEntry($file).Open())
        try{if($reader.ReadToEnd().Contains('private-existing')){throw 'User workflow entered ZIP'}}finally{$reader.Dispose()}
    }
    if('pari-data/seadata/sea0' -notin $names -or 'licenses/PARI-GPL-2.0.txt' -notin $names){throw 'Nested PARI dependency missing'}
}finally{$zip.Dispose()}
$rejected=$false
try{& (Join-Path $repo 'tools/build/internal/package_archive.ps1') -PackageDir $package -BuildDir $Output -Stage stage2 -Files @('../escape.dll')}catch{$rejected=$true}
if(-not $rejected){throw 'Archive path escape accepted'}
$bad=Join-Path $Output 'tampered-build';New-Item -ItemType Directory -Path $bad | Out-Null
Copy-Item -LiteralPath (Join-Path $Build 'diagnostics_build_manifest.json') -Destination $bad
[IO.File]::WriteAllText((Join-Path $bad 'ecm_diagnostics_host.exe'),'tampered-host')
$rejected=$false
try{Copy-EcmDiagnosticsPayload $repo $bad (Join-Path $Output 'rejected') | Out-Null}catch{$rejected=$true}
if(-not $rejected){throw 'Tampered diagnostic host accepted'}
[ordered]@{passed=$true;files=$files.Count;zip_relative_directories=$true;user_config_excluded=$true;path_escape_rejected=$true;tampered_host_rejected=$true} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output 'result.json') -Encoding UTF8
Write-Host 'PASS: diagnostic dependency ZIP, workflow exclusion, path confinement and frozen identity'
