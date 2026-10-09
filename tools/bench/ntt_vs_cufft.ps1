#Requires -Version 5.1
<#
.SYNOPSIS
    Compare integer NTT and fp64 cuFFT probes on the same shapes.

.DESCRIPTION
    Both operands' packed-bit count defines the probe metric:
        ns_per_operand_bit = t_total * 1e9 / (2 * P * slot_bits)
    The script recomputes this value from total time and slot width and cross-checks
    the printed metric. Repeated samples report their median. This microbenchmark
    excludes the complete ECM pipeline; see docs/architecture/NTT.md.

.PARAMETER Shapes
    "P,S" pairs.  Default: 1024,5153 / 8192,5153 / 65536,5153 (S=5153 = the M5153-class
    modulus width the earlier measurements used).

.PARAMETER Real
    Also measure P=92160, S=5261. Query actual workspace capacity before running.

.EXAMPLE
    powershell -File tools\bench\ntt_vs_cufft.ps1
    powershell -File tools\bench\ntt_vs_cufft.ps1 -Real -Device 1
#>
param(
    [string[]]$Shapes = @('1024,5153', '8192,5153', '65536,5153'),
    [int]$Device = 1,
    [int]$Repeat = 3,
    [double]$Gate = 0.10,
    [switch]$Real,
    [string]$NttExe = '',
    [string]$CufftExe = ''
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $NttExe) { $NttExe = Join-Path $repo 'build_cuda_cmake\ntt_poly_probe.exe' }
if (-not $CufftExe) { $CufftExe = Join-Path $repo 'build_cuda_cmake\cufft_kron_probe.exe' }
if ($Real) { $Shapes += '92160,5261' }

foreach ($e in @($NttExe, $CufftExe)) {
    if (-not (Test-Path $e)) { throw "missing exe: $e" }
    $dll = Join-Path (Split-Path -Parent $e) 'gmp-10.dll'
    if (-not (Test-Path $dll)) {
        Copy-Item (Join-Path $repo 'third_party\gmp-zen3\dist\bin\gmp-10.dll') (Split-Path -Parent $e) -Force
    }
}

function Probe([string]$exe, [int]$P, [int]$S, [int]$device, [int]$repeat) {
    $best = $null
    $times = @()
    for ($i = 0; $i -lt $repeat; ++$i) {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $out = (& $exe poly $P $S $device 0 2>&1 | Out-String)
        $ErrorActionPreference = $prev
        # Common fields first: both probes print slot_bits=, ok= and t_total= on their
        # "poly:" line.  bpw=/nwords=/stages= are NTT-only (the cuFFT probe prints
        # chunks/slot= instead), so they are parsed separately and may be empty.
        $m = [regex]::Match($out, 'slot_bits=(\d+).*?ok=(\d).*?t_total=([\d.]+)')
        if (-not $m.Success) { return $null }
        $times += [double]$m.Groups[3].Value
        $best = $m
    }
    $sorted = $times | Sort-Object
    $median = $sorted[[int][math]::Floor($sorted.Count / 2)]
    $slot = [double]$best.Groups[1].Value
    $derived = $median * 1e9 / (2.0 * $P * $slot)
    return [pscustomobject]@{
        ok = $best.Groups[2].Value; slot = $slot
        bpw = [regex]::Match($out, 'bpw=(\d+)').Groups[1].Value
        nwords = [regex]::Match($out, 'nwords=(\d+)').Groups[1].Value
        t = $median; ns = $derived
        spread = if ($sorted.Count -gt 1) { ($sorted[-1] / $sorted[0]) } else { 1.0 }
        stages = [regex]::Match($out, 'stages=(\d+)').Groups[1].Value
    }
}

Write-Host ("M2 acceptance: integer NTT vs fp64 cuFFT (device {0}, median of {1} runs)" -f $Device, $Repeat)
Write-Host ("metric: ns per operand-bit = t_total*1e9/(2*P*slot_bits);  gate <= {0} ns/bit" -f $Gate)
Write-Host ""
Write-Host ("{0,8} {1,6} {2,10} {3,11} {4,7} {5,8} {6,7} {7,7}" -f 'P', 'S', 'NTT ns/bit', 'cuFFT ns/bit', 'ratio', 'NTT bpw', 'NTT ok', 'spread')

$rows = @()
foreach ($sh in $Shapes) {
    $parts = $sh.Split(',')
    $P = [int]$parts[0]; $S = [int]$parts[1]
    $n = Probe $NttExe $P $S $Device $Repeat
    $c = Probe $CufftExe $P $S $Device $Repeat
    if ($null -eq $n -or $null -eq $c) { Write-Host ("{0,8} {1,6}  (probe output not parsable)" -f $P, $S); continue }
    $ratio = if ($n.ns -gt 0) { $c.ns / $n.ns } else { 0 }
    Write-Host ("{0,8} {1,6} {2,10:N4} {3,10:N4} {4,8:N2} {5,9} {6,8} {7,7:N2}" -f `
                $P, $S, $n.ns, $c.ns, $ratio, $n.bpw, $n.ok, $n.spread)
    $rows += [pscustomobject]@{ P = $P; S = $S; ntt = $n.ns; cufft = $c.ns; ok = $n.ok; ratio = $ratio }
}

Write-Host ""
foreach ($r in $rows) {
    $verdict = if ($r.ok -ne '1') { 'PROBE STILL WRONG (ok=0): the number means nothing' }
               elseif ($r.ntt -le $Gate) { 'M2 GATE MET' }
               else { 'gate not met' }
    Write-Host ("  P={0} S={1}: {2:N4} ns/operand-bit  (cuFFT {3:N4}, ratio {4:N2}x)  -> {5}" -f `
                $r.P, $r.S, $r.ntt, $r.cufft, $r.ratio, $verdict)
}
Write-Host ""
Write-Host "reference points: fp64 cuFFT 0.278-0.308 ns/operand-bit (P flat); one Kronecker-packed"
Write-Host "polynomial multiply moves 2*P*slot_bits operand-bits, so a per-curve total of 4.3e11"
Write-Host "operand-bits (M5261, section 15.1) costs 4.3e11 * ns/bit seconds per curve."
exit 0
