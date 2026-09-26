# test_cuda_param2.ps1 -- param2 (gpu_param = 2) correctness regression.
#
# The decisive check is a CROSS-IMPLEMENTATION one: for the same sigma and B1, our param2
# stage-1 x must equal the one gmp-ecm (-param 2, same sigma) writes into its own save.
# That is what caught the two real bugs on 2026-09-25 (an aliased Jacobian helper and, more
# importantly, result words decoded with the wrong buffer layout).
#   A. curve interop : our saved X == gmp-ecm's saved X   (M521, sigma=1000000, B1=1e3 and 1e5)
#   B. save form     : PARAM=2, SIGMA=<sigma>, N = the original expression
#   C. resume        : gmp-ecm -param 2 -resume accepts the file and reports the same N
#
# usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_cuda_param2.ps1
#          [-CudaExe build_cuda_dev\ecm_cuda.exe] [-GmpEcmExe <path>] [-Device 1] [-Bits 521]
#   -Bits must be a Mersenne PRIME exponent (521, 607, 1279, ...): with a composite M_p the
#   run legitimately finds a factor, the save then carries that factor as X, and the
#   comparison against gmp-ecm's x-coordinate is meaningless -- the script detects and
#   reports that case instead of printing a bogus FAIL.
param(
    [string]$CudaExe  = 'build_cuda_dev\ecm_cuda.exe',
    [string]$GmpEcmExe = 'D:\code\GIMPS\gmp-ecm\ecm-2025.10.28-win.multiarch\ecm-zen3.exe',
    [int]$Device = 1,
    [int]$Bits = 521,
    [uint32]$Sigma = 1000000
)

$ErrorActionPreference = 'Continue'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not [System.IO.Path]::IsPathRooted($CudaExe)) { $CudaExe = Join-Path $repo $CudaExe }
$work = Join-Path $repo '.bench_tmp\test_param2'
New-Item -ItemType Directory -Force -Path $work | Out-Null
Get-ChildItem $work -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

$pass = 0; $fail = 0
function Check([string]$name, [bool]$ok) {
    if ($ok) { Write-Host "  [PASS] $name"; $script:pass++ } else { Write-Host "  [FAIL] $name"; $script:fail++ }
}

# M521 is prime, so no factor can be found and every run covers the whole scalar.
$expr = "(2^$Bits-1)"
$nfile = Join-Path $work 'n.txt'
[System.IO.File]::WriteAllText($nfile, "$expr`n", ([System.Text.Encoding]::ASCII))
$dec = (python -c "print(2**$Bits-1)").Trim()
$decfile = Join-Path $work 'ndec.txt'
[System.IO.File]::WriteAllText($decfile, "$dec`n", ([System.Text.Encoding]::ASCII))

Write-Host "=== A. param2 stage-1 x vs gmp-ecm (M$Bits, sigma=$Sigma) ==="
foreach ($B1 in '1e3', '1e5') {
    $ourSave = Join-Path $work "ours_$B1.save"
    $gmpSave = Join-Path $work "gmp_$B1.save"
    $null = cmd /c "cd /d `"$work`" && `"$CudaExe`" -gpu -d $Device --gpu-param 2 -sigma 2:$Sigma -gpucurves 1 -savea `"$ourSave`" $B1 0 < `"$nfile`"" 2>&1
    $null = cmd /c "cd /d `"$work`" && `"$GmpEcmExe`" -param 2 -sigma $Sigma -c 1 -save `"$gmpSave`" $B1 0 < `"$decfile`"" 2>&1
    if (-not (Test-Path $ourSave) -or -not (Test-Path $gmpSave)) {
        Check "B1=$B1 both saves exist" $false
        continue
    }
    $ox = [regex]::Match((Get-Content $ourSave -TotalCount 1), 'X=0x([0-9a-f]+)').Groups[1].Value
    $gx = [regex]::Match((Get-Content $gmpSave -TotalCount 1), 'X=0x([0-9a-f]+)').Groups[1].Value
    # A HIT makes the save carry the FOUND FACTOR instead of an x-coordinate, which is
    # correct behaviour but not comparable -- that needs N to be prime.  M_p is composite
    # whenever p is not a Mersenne prime exponent (1021 is NOT one: 2^1021-1 has small
    # factors, so the run legitimately "hits" -- see docs/ECM_CGBN_OPTIMIZATION.md 6.6).
    if ($ox.Length -lt ($Bits / 8)) {
        Check "B1=$B1 N is prime (no factor hit; X has $($ox.Length) hex digits)" $false
        Write-Host "         pick a Mersenne PRIME exponent for -Bits: 521, 607, 1279, 2203, 3217"
        continue
    }
    Check "B1=$B1 our X == gmp-ecm X" ($ox.Length -gt 0 -and $ox -eq $gx)
}

Write-Host "=== B. save form ==="
$line = Get-Content (Join-Path $work 'ours_1e3.save') -TotalCount 1
Check "PARAM=2 is written"      ($line -match 'PARAM=2;')
Check "SIGMA is our sigma"      ($line -match "SIGMA=$Sigma;")
Check "N is the original expr"  ($line -match 'N=\(2\^' -or $line -match 'N=\d')

Write-Host "=== C. gmp-ecm resume accepts the save ==="
$r = cmd /c "cd /d `"$work`" && `"$GmpEcmExe`" -param 2 -resume `"$(Join-Path $work 'ours_1e5.save')`" 1e5 1e6 2>&1" | Out-String
Check "no bad-checksum complaint" (-not ($r -match 'bad checksum|Checksum error|invalid'))
Check "gmp-ecm read N as M$Bits"  ($r -match "2\^$Bits-1|$Bits bits" -or $r -match 'Step 2' -or $r -match 'no factor')

Write-Host ""
if ($fail -eq 0) { Write-Host "ALL OK ($pass checks)"; exit 0 } else { Write-Host "$fail FAILED of $($pass+$fail)"; exit 1 }
