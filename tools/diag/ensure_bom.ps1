<#
.SYNOPSIS
    UTF-8 BOM guard for source files: nvcc reads a BOM-less file as GBK (code page 936).

.DESCRIPTION
    nvcc's frontend reads a source file that has non-ASCII bytes as ANSI unless the file
    starts with a UTF-8 BOM.  A CJK comment then ends in a lead byte that swallows the
    newline, which comments out the NEXT line of code.  That trap is recorded in
    docs/performance/STAGE1.md 6, and in the Mersenne-fold round it bit again: a file
    edit round-trip dropped the BOM of kernels/cuda/cgbn_stage1.cu and silently removed
    `#define CHECKPOINT_VERSION` (the build then failed with "identifier is undefined").

    This guard compares existing source files against Git HEAD and restores their
    BOM state. It reads Git blobs in one batch and checks the working file
    prefixes before writing any corrections.

.PARAMETER Path
    Directories to scan (default: kernels, src).  Recurses.

.PARAMETER NoFix
    Report only; do not touch any file.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\diag\ensure_bom.ps1
#>
param(
    [string[]]$Path = @("kernels", "src"),
    [switch]$NoFix
)

$ErrorActionPreference = 'Stop'
$guardArgs = @()
if ($NoFix) { $guardArgs += '--no-fix' }
& python (Join-Path $PSScriptRoot '../build/internal/ensure_bom.py') @guardArgs -- @Path
exit $LASTEXITCODE
