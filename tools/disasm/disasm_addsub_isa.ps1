param(
    [int]$Bits = 4096
)

$ErrorActionPreference = "Stop"

$root = "D:\code\MPA-OpenCl"
Set-Location $root

$exportExe = ".\build\Debug\opencl_addsub_isa_export.exe"
$binPath = "bench\addsub_isa_${Bits}_amd.bin"
$isaStem = "bench\addsub_isa_${Bits}_amd.rga.isa.txt"
$analysisStem = "bench\addsub_isa_${Bits}_amd.rga.analysis.csv"
$liveregStem = "bench\addsub_isa_${Bits}_amd.rga.livereg.txt"

if (!(Test-Path $exportExe)) {
    throw "Missing exporter: $exportExe (build target opencl_addsub_isa_export first)."
}

$rga = Get-Command rga -ErrorAction SilentlyContinue
if ($null -eq $rga) {
    throw "rga not found in PATH."
}

Write-Host "[1/2] Exporting ${Bits}-bit add/sub/mod pure kernels..."
& $exportExe --bits $Bits --no-list
if (!(Test-Path $binPath)) {
    throw "Exported binary not found: $binPath"
}

Write-Host "[2/2] Running RGA bin analysis..."
& $rga.Source -s bin --co $binPath --isa $isaStem --analysis $analysisStem --livereg $liveregStem

$kernels = @(
    "ecm_mp_add_n",
    "ecm_mp_sub_n",
    "ecm_mp_add_mod_fused",
    "ecm_mp_add_mod_legacy",
    "ecm_mp_add_mod_mask",
    "ecm_mp_sub_mod"
)

Write-Host ""
Write-Host "Generated per-kernel artifacts (gfx1150 prefix may vary by GPU):"
foreach ($k in $kernels) {
    $isa = Get-ChildItem -Path bench -Filter "gfx*_${k}_addsub_isa_${Bits}_amd.rga.isa.txt" -ErrorAction SilentlyContinue
    $livereg = Get-ChildItem -Path bench -Filter "gfx*_${k}_addsub_isa_${Bits}_amd.rga.livereg.txt" -ErrorAction SilentlyContinue
    $analysis = Get-ChildItem -Path bench -Filter "gfx*_${k}_addsub_isa_${Bits}_amd.rga.analysis.csv" -ErrorAction SilentlyContinue
    if ($isa) { Write-Host "  ISA: $($isa.Name)" }
    if ($livereg) { Write-Host "  LiveReg: $($livereg.Name)" }
    if ($analysis) { Write-Host "  Analysis: $($analysis.Name)" }
}
