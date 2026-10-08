# Resolve one coherent GMP header/import-library/runtime set.
function Resolve-EcmGmp([string]$Repo, [string]$Gmp, [switch]$Release) {
    if (-not $Gmp) {
        if ($Release) {
            $candidates = @()
            foreach ($variable in @('VCPKG_ROOT','VCPKG_INSTALLATION_ROOT')) {
                $value = [Environment]::GetEnvironmentVariable($variable)
                if ($value) { $candidates += Join-Path $value 'installed/x64-windows' }
            }
            $candidates += @(Join-Path (Split-Path -Parent $Repo) 'vcpkg/installed/x64-windows')
            $candidates += @('D:/code/vcpkg/installed/x64-windows','C:/code/vcpkg/installed/x64-windows')
            $Gmp = $candidates | Where-Object { Test-Path -LiteralPath (Join-Path $_ 'bin/gmp-10.dll') } | Select-Object -First 1
            if (-not $Gmp) { throw 'Release GMP not found; set VCPKG_ROOT or pass -Gmp <x64-windows prefix>. No Zen3 fallback.' }
        } else { $Gmp = Join-Path $Repo 'third_party/gmp-zen3/dist' }
    }
    if (-not [IO.Path]::IsPathRooted($Gmp)) { $Gmp = Join-Path $Repo $Gmp }
    $Gmp = [IO.Path]::GetFullPath($Gmp)
    $hashes = [ordered]@{}
    foreach ($name in @('include/gmp.h','lib/gmp.lib','bin/gmp-10.dll')) {
        $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $Gmp $name) -Algorithm SHA256).Hash
    }
    $copyright = Join-Path $Gmp 'share/gmp/copyright'
    if ($Release -and -not (Test-Path -LiteralPath $copyright)) { throw "Release GMP copyright file missing: $copyright" }
    return [pscustomobject]@{ Root = $Gmp; Hashes = $hashes; Copyright = $copyright }
}
