#Requires -Version 5.1
<#
.SYNOPSIS
    Build the Windows x64 GWNUM Stage1 plugin for local use.
.PARAMETER GwNumRoot
    Directory containing the official GWNUM 31.6 headers and gwnum64.lib.
    If the library is absent, build the upstream gwnum64.lib target here.
.PARAMETER BuildDir
    Directory for the ECM-ELF plugin build. Relative paths are resolved from the repository.
#>
param(
    [string]$GwNumRoot = '.refactor/p95v3106b01.source/gwnum',
    [string]$BuildDir = 'build_dev/gwnum_local'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'internal/build_directory.ps1')
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (-not [IO.Path]::IsPathRooted($GwNumRoot)) { $GwNumRoot = Join-Path $repo $GwNumRoot }
if (-not [IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
$GwNumRoot = [IO.Path]::GetFullPath($GwNumRoot)
$BuildDir = [IO.Path]::GetFullPath($BuildDir)

if (-not (Get-Command cl.exe -CommandType Application -ErrorAction SilentlyContinue) -or
    -not (Get-Command nmake.exe -CommandType Application -ErrorAction SilentlyContinue) -or
    $env:VSCMD_ARG_TGT_ARCH -ne 'x64') {
    throw 'Run this script from the Visual Studio x64 Native Tools command prompt or x64 Developer PowerShell.'
}
$cmake = (Get-Command cmake.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$header = Join-Path $GwNumRoot 'gwnum.h'
if (-not (Test-Path -LiteralPath $header -PathType Leaf)) {
    throw "GWNUM 31.6 headers not found at $GwNumRoot. Extract the Prime95 31.6 source archive or pass -GwNumRoot."
}
if (-not (Select-String -LiteralPath $header -Pattern '^\s*#\s*define\s+GWNUM_VERSION\s+"31\.6"' -Quiet)) {
    throw "GWNUM 31.6 is required: $header"
}

$library = Join-Path $GwNumRoot 'gwnum64.lib'
if (-not (Test-Path -LiteralPath $library -PathType Leaf)) {
    if (-not (Test-Path -LiteralPath (Join-Path $GwNumRoot 'compil64') -PathType Leaf)) {
        throw "No gwnum64.lib or upstream compil64 makefile at $GwNumRoot."
    }
    [void][IO.Directory]::CreateDirectory((Join-Path $GwNumRoot 'amd64/release'))
    Push-Location -LiteralPath $GwNumRoot
    try {
        & nmake.exe /f compil64 gwnum64.lib
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $library -PathType Leaf)) {
            throw 'GWNUM build failed. If the assembler was requested, make upstream UASM64 available on PATH.'
        }
    } finally {
        Pop-Location
    }
}

Initialize-EcmBuildDirectory $BuildDir
& $cmake -S (Join-Path $PSScriptRoot 'gwnum') -B $BuildDir -G 'NMake Makefiles' '-DCMAKE_BUILD_TYPE=Release' "-DECM_GWNUM_ROOT=$GwNumRoot"
if ($LASTEXITCODE -ne 0) { throw 'GWNUM DLL configuration failed.' }
& $cmake --build $BuildDir --target ecm_gwnum
if ($LASTEXITCODE -ne 0) { throw 'GWNUM DLL build failed.' }

$dll = Join-Path $BuildDir 'gwnum.dll'
if (-not (Test-Path -LiteralPath $dll -PathType Leaf)) { throw "Build output missing: $dll" }
Write-Host "Built $dll"
Write-Host 'Copy gwnum.dll beside ecm.exe or ecm_cuda.exe and select --impl gwnum.'
