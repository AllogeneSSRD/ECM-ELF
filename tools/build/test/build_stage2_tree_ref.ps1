#Requires -Version 5.1
<#
.SYNOPSIS
    Build tools/bench/stage2_tree_ref.cpp (the TREE stage-2 reference) with MSVC + the
    repo's GMP.

.DESCRIPTION
    Host-only translation unit: no nvcc, no CGBN, no CUDA.  It is deliberately NOT part
    of the CMake target list (like the other tools/bench probes), so the CMake build is
    untouched and no reconfigure is ever needed -- a normal MSVC compile of one file plus
    a link against third_party/gmp-zen3/dist/lib/gmp.lib.

    gmp-10.dll has to sit next to the exe (the same trap the other bench probes document);
    it is copied automatically.

.EXAMPLE
    powershell -File tools\build\test\build_stage2_tree_ref.ps1
#>
param(
    [string]$Build = 'build_cuda_cmake'
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $repo

$vcvars = (Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\*\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue |
           Select-Object -First 1).FullName
if (-not $vcvars) { throw "vcvars64.bat not found; install the MSVC build tools" }

$exe = Join-Path $Build 'stage2_tree_ref.exe'
New-Item -ItemType Directory -Force $Build | Out-Null

# NOTE the object path is a FULL FILE NAME, not a directory with a trailing backslash:
# `/Fo:"dir\"` is the classic Windows trap -- cmd reads the `\"` as an escaped quote, the
# quoted region swallows the rest of the command line and cl reports
# "cannot open compiler generated file: <dir>\<rest of the line>" (measured).
$objDir = Join-Path $Build '_stage2tree'
New-Item -ItemType Directory -Force $objDir | Out-Null
$obj = Join-Path $objDir 'stage2_tree_ref.obj'

$line = "call `"$vcvars`" >nul 2>&1 && cl /nologo /O2 /std:c++17 /EHsc " +
        "/I third_party\gmp-zen3\dist\include tools\bench\stage2_tree_ref.cpp " +
        "/Fe:`"$exe`" /Fo:`"$obj`" " +
        "/link /LIBPATH:third_party\gmp-zen3\dist\lib gmp.lib"

Write-Host ("stage2_tree_ref build: exe={0}" -f $exe)
& cmd.exe /c $line
if ($LASTEXITCODE -ne 0) { throw "compile/link failed (exit $LASTEXITCODE)" }

$gmpDll = 'third_party\gmp-zen3\dist\bin\gmp-10.dll'
if (Test-Path $gmpDll) { Copy-Item $gmpDll (Split-Path -Parent $exe) -Force }

Write-Host ("built {0} ({1:N0} bytes)" -f $exe, (Get-Item $exe).Length)
exit 0
