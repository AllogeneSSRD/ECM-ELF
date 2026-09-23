$ErrorActionPreference = "Stop"

$root = "D:\code\MPA-OpenCl"
Set-Location $root

$exportExe = ".\build\Debug\opencl_mont_isa_export.exe"
$binPath = "bench\mont_isa_4096_amd.bin"
$readobjTxt = "bench\mont_isa_4096_amd.readobj.txt"
$objdumpTxt = "bench\mont_isa_4096_amd.objdump.txt"
$summaryCsv = "bench\mont_isa_4096_amd.symbol_summary.csv"

if (!(Test-Path $exportExe)) {
    throw "Missing exporter executable: $exportExe (build target opencl_mont_isa_export first)."
}

$llvmReadobj = (Get-Command llvm-readobj -ErrorAction SilentlyContinue)
if ($null -eq $llvmReadobj) {
    throw "llvm-readobj not found in PATH."
}

$llvmObjdump = (Get-Command llvm-objdump -ErrorAction SilentlyContinue)

Write-Host "[1/4] Exporting 4096-bit mont AMD binary..."
& $exportExe

if (!(Test-Path $binPath)) {
    throw "Exported binary not found: $binPath"
}

Write-Host "[2/4] Dumping ELF headers/sections/symbols via llvm-readobj..."
& $llvmReadobj.Source --file-headers --sections --symbols $binPath | Out-File -FilePath $readobjTxt -Encoding ascii

Write-Host "[3/4] Parsing symbol sizes for mont kernels..."
$content = Get-Content $readobjTxt

$targets = @(
    "ecm_mont_mul_priv_bench",
    "ecm_mont_sqr_priv_bench",
    "cgbn_mont_mul_wg_bench",
    "cgbn_mont_sqr_wg_bench",
    "cgbn_mont_mul_wg",
    "cgbn_mont_sqr_wg"
)

"symbol,size_bytes,inst_count_min_est,inst_count_max_est" | Out-File -FilePath $summaryCsv -Encoding ascii

for ($i = 0; $i -lt $content.Count; ++$i) {
    $line = $content[$i]
    foreach ($sym in $targets) {
        if ($line -match "Name:\s+$([regex]::Escape($sym))\b") {
            $size = -1
            $isFunction = $false
            for ($j = $i; $j -lt [Math]::Min($i + 12, $content.Count); ++$j) {
                if ($content[$j] -match "Size:\s+([0-9]+)") {
                    $size = [int]$Matches[1]
                }
                if ($content[$j] -match "Type:\s+Function") {
                    $isFunction = $true
                }
            }
            if (-not $isFunction) {
                continue
            }
            if ($size -ge 0) {
                # AMD GCN instructions are variable 4/8 bytes, so provide interval estimate.
                $minInst = [int][Math]::Floor($size / 8.0)
                $maxInst = [int][Math]::Floor($size / 4.0)
                "$sym,$size,$minInst,$maxInst" | Out-File -FilePath $summaryCsv -Append -Encoding ascii
            }
        }
    }
}

Write-Host "[4/4] Trying llvm-objdump disassembly (best effort)..."
if ($null -ne $llvmObjdump) {
    & $llvmObjdump.Source -d $binPath | Out-File -FilePath $objdumpTxt -Encoding ascii
    if ($LASTEXITCODE -eq 0) {
        Write-Host "objdump written: $objdumpTxt"
    } else {
        Write-Host "llvm-objdump failed for AMDGCN target on this LLVM build."
        Write-Host "Install RGA (recommended) or LLVM with AMDGPU backend."
        if (Test-Path $objdumpTxt) {
            Remove-Item $objdumpTxt -ErrorAction SilentlyContinue
        }
        $global:LASTEXITCODE = 0
    }
} else {
    Write-Host "llvm-objdump missing; skip disassembly."
}

Write-Host ""
Write-Host "Generated:"
Write-Host "  $readobjTxt"
Write-Host "  $summaryCsv"
if (Test-Path $objdumpTxt) { Write-Host "  $objdumpTxt" }
