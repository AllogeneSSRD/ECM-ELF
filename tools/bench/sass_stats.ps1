# sass_stats.ps1 -- export SASS for one kernel instantiation and report instruction statistics.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools\bench\sass_stats.ps1
#          [-Obj <path to .obj>] [-Kernel <substring of the mangled name>] [-Top 12]
# Purpose (2026-09-25): check whether a register-capped build pays for occupancy with extra
# instructions (spills = LDL/STL) and see the inner-loop opcode mix, per
# docs/ECM_CGBN_OPTIMIZATION.md section 8.
param(
    [string]$Obj = 'build_cuda_cmake\CMakeFiles\ecm_cuda.dir\kernels\cuda\cgbn_stage1_kernels_tpi16.cu.obj',
    [string]$Kernel = 'kernel_double_addI13cgbn_params_tILj16ELj8192E',
    [int]$Top = 12
)
$ErrorActionPreference = 'Continue'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not [System.IO.Path]::IsPathRooted($Obj)) { $Obj = Join-Path $repo $Obj }
if (-not (Test-Path $Obj)) { Write-Host "no such object: $Obj"; exit 2 }

$dump = Join-Path $repo '.bench_tmp\sass_dump.txt'
cmd /c "cuobjdump -sass `"$Obj`" > `"$dump`" 2>&1" | Out-Null
if (-not (Test-Path $dump)) { Write-Host 'cuobjdump failed'; exit 2 }

$lines = Get-Content $dump
$start = -1
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match 'Function : ' -and $lines[$i] -match [regex]::Escape($Kernel)) { $start = $i; break }
}
if ($start -lt 0) {
    Write-Host "kernel not found for '$Kernel'; available functions:"
    $lines | Select-String -Pattern 'Function : ' | Select-Object -First 8 | ForEach-Object { '   ' + $_.Line.Trim() }
    exit 3
}
$end = $lines.Count
for ($i = $start + 1; $i -lt $lines.Count; $i++) { if ($lines[$i] -match 'Function : ') { $end = $i; break } }

$body = $lines[($start + 1)..($end - 1)] | Where-Object { $_ -match '/\*[0-9a-f]{4,}\*/' }
$ops = @{}
foreach ($l in $body) {
    $m = [regex]::Match($l, '\*/\s+@?!?P?\d*\s*([A-Z][A-Z0-9._]+)')
    if ($m.Success) {
        $op = $m.Groups[1].Value -replace '\..*$', ''
        $ops[$op] = 1 + ($ops[$op] | ForEach-Object { $_ })
    }
}
$total = ($ops.Values | Measure-Object -Sum).Sum
Write-Host ("kernel : {0}" -f $Kernel)
Write-Host ("object : {0}" -f (Split-Path $Obj -Leaf))
Write-Host ("SASS instructions: {0}" -f $total)
Write-Host ("spill traffic    : LDL={0} STL={1}" -f ($ops['LDL'] | ForEach-Object { $_ }), ($ops['STL'] | ForEach-Object { $_ }))
Write-Host "top opcodes:"
$ops.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First $Top | ForEach-Object {
    Write-Host ("   {0,-10} {1,6}  ({2,5:N1}%)" -f $_.Key, $_.Value, (100.0 * $_.Value / [Math]::Max($total, 1)))
}
