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

    This guard compares every source file with the version in git HEAD and restores the
    BOM state HEAD has.  Only files git tracks are considered, so a deliberately BOM-less
    file (most of src/) is left alone -- the rule is "same BOM as HEAD", not "always BOM".

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

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$bom = [byte[]](0xEF, 0xBB, 0xBF)
$fixedBom = 0; $fixedNoBom = 0; $scanned = 0
$tmp = Join-Path $env:TEMP "ensure_bom_head.bin"

foreach ($p in $Path) {
    $root = if ([System.IO.Path]::IsPathRooted($p)) { $p } else { Join-Path $repo $p }
    if (-not (Test-Path $root)) { continue }
    foreach ($f in (Get-ChildItem -Path $root -Recurse -File -Include '*.cu', '*.cuh', '*.h', '*.hpp', '*.cpp', '*.cc' -ErrorAction SilentlyContinue)) {
        $rel = $f.FullName.Substring($repo.Length).TrimStart('\').Replace('\', '/')
        $scanned++
        if (Test-Path $tmp) { Remove-Item $tmp -Force }
        cmd /c "git cat-file blob HEAD:`"$rel`" > `"$tmp`" 2>nul" | Out-Null
        if (-not (Test-Path $tmp) -or (Get-Item $tmp).Length -eq 0) { continue }   # untracked / new file
        $hb = [System.IO.File]::ReadAllBytes($tmp)
        $wb = [System.IO.File]::ReadAllBytes($f.FullName)
        if ($hb.Length -lt 3 -or $wb.Length -lt 3) { continue }
        $hbom = ($hb[0] -eq 0xEF -and $hb[1] -eq 0xBB -and $hb[2] -eq 0xBF)
        $wbom = ($wb[0] -eq 0xEF -and $wb[1] -eq 0xBB -and $wb[2] -eq 0xBF)
        if ($hbom -eq $wbom) { continue }
        if ($NoFix) {
            Write-Host "  [BOM] $rel differs from HEAD (HEAD bom=$hbom, working bom=$wbom)" -ForegroundColor Yellow
        } elseif ($hbom) {
            [System.IO.File]::WriteAllBytes($f.FullName, ($bom + $wb))
            Write-Host "  [BOM] restored UTF-8 BOM in $rel (nvcc would read it as GBK)" -ForegroundColor Yellow
            $fixedBom++
        } else {
            [System.IO.File]::WriteAllBytes($f.FullName, $wb[3..($wb.Length - 1)])
            Write-Host "  [BOM] removed BOM from $rel (HEAD has none -- keeps the diff clean)" -ForegroundColor Yellow
            $fixedNoBom++
        }
    }
}
if (Test-Path $tmp) { Remove-Item $tmp -Force }
Write-Host "  BOM guard: $scanned file(s) scanned, $fixedBom restored, $fixedNoBom cleared"
exit 0
