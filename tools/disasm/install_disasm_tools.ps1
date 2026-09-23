$ErrorActionPreference = "Stop"

function Has-Command([string]$Name) {
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Info([string]$Msg) {
    Write-Host "[INFO] $Msg"
}

function Warn([string]$Msg) {
    Write-Host "[WARN] $Msg" -ForegroundColor Yellow
}

Info "Installing disassembly toolchain for AMD OpenCL analysis."
Info "Targets: LLVM (llvm-objdump/llvm-readobj), Radeon GPU Analyzer (rga)."

if (-not (Has-Command "winget")) {
    throw "winget not found. Please install App Installer from Microsoft Store first."
}

# 1) LLVM via winget (for llvm-objdump/llvm-readobj)
if (Has-Command "llvm-objdump") {
    Info "llvm-objdump already available."
} else {
    Info "Installing LLVM via winget (LLVM.LLVM)..."
    winget install --id LLVM.LLVM --accept-package-agreements --accept-source-agreements --silent
}

# 2) Try Radeon GPU Analyzer via winget if available
$rgaInstalled = $false
if (Has-Command "rga") {
    Info "rga already available."
    $rgaInstalled = $true
} else {
    Info "Attempting to install Radeon GPU Analyzer via winget search/install..."
    $searchOut = winget search "Radeon GPU Analyzer" | Out-String
    if ($searchOut -match "Radeon GPU Analyzer") {
        try {
            # ID may vary over time; best effort install by name.
            winget install --name "Radeon GPU Analyzer" --accept-package-agreements --accept-source-agreements --silent
            if (Has-Command "rga") { $rgaInstalled = $true }
        } catch {
            Warn "winget install for RGA failed: $($_.Exception.Message)"
        }
    } else {
        Warn "RGA not found in winget source."
    }
}

# 3) Refresh PATH in current shell for common LLVM install locations
$llvmCandidates = @(
    "C:\Program Files\LLVM\bin",
    "C:\Program Files (x86)\LLVM\bin"
)
foreach ($p in $llvmCandidates) {
    if ((Test-Path $p) -and ($env:Path -notlike "*$p*")) {
        $env:Path = "$p;$env:Path"
    }
}

# 4) Final status
Info "Final tool detection:"
$llvmObjdump = Get-Command llvm-objdump -ErrorAction SilentlyContinue
$llvmReadobj = Get-Command llvm-readobj -ErrorAction SilentlyContinue
$rgaCmd = Get-Command rga -ErrorAction SilentlyContinue

if ($llvmObjdump) {
    Info "llvm-objdump: $($llvmObjdump.Source)"
} else {
    Warn "llvm-objdump NOT found."
}

if ($llvmReadobj) {
    Info "llvm-readobj: $($llvmReadobj.Source)"
} else {
    Warn "llvm-readobj NOT found."
}

if ($rgaCmd) {
    Info "rga: $($rgaCmd.Source)"
    $rgaInstalled = $true
} else {
    Warn "rga NOT found."
}

if (-not $rgaInstalled) {
    Warn "Please install RGA manually from AMD GPUOpen:"
    Warn "https://gpuopen.com/rga/"
}

Info "Done."
