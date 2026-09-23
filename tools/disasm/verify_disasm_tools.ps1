$ErrorActionPreference = "Stop"

function Show-Cmd([string]$name) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) {
        Write-Host "[OK] $name -> $($cmd.Source)"
        return $true
    }
    Write-Host "[MISS] $name"
    return $false
}

Write-Host "Checking disassembly tools..."
$ok1 = Show-Cmd "llvm-objdump"
$ok2 = Show-Cmd "llvm-readobj"
$ok3 = Show-Cmd "rga"

Write-Host ""
Write-Host "Suggested next commands:"
Write-Host "  .\build\Debug\opencl_mont_isa_export.exe"
Write-Host "  # then disassemble exported binary with rga/llvm tools"

if (-not ($ok1 -or $ok3)) {
    Write-Host ""
    Write-Host "No usable disasm tool found. Run:"
    Write-Host "  .\tools\disasm\install_disasm_tools.ps1"
}
