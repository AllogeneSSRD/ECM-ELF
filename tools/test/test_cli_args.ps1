# test_cli_args.ps1 -- CLI argument semantics for the parametrization selectors.
#
# gmp-ecm documents two spellings for the curve parameter:
#   -param  i     which parametrization should be used
#   -sigma  s     use s as parameter to compute the curve's coefficients
#                 can use -sigma i:s to specify -param i at the same time
# so "-sigma i:s" is a parametrization CLAIM.  Combining it with a *different* explicit
# selector must fail loudly instead of silently letting one of them win:
#   Error, conflict between -sigma and -param arguments
# In this repository the explicit selector is --gpu-param (0 = Suyama, 2 = batch 2 / 6-torsion,
# 3 = gmp-ecm batch, the default).
#
# Checks:
#   1. --gpu-param 0 -sigma 3:<s>   -> rejected, message mentions the conflict
#   2. --gpu-param 2 -sigma 3:<s>   -> rejected
#   3. --gpu-param 3 -sigma 2:<s>   -> rejected
#   4. --gpu-param 3 -sigma 3:<s>   -> accepted (agreeing prefixes are fine)
#   5. --gpu-param 0 -sigma <s>     -> accepted (a bare sigma makes no param claim)
#   6. -sigma 3:<s> alone           -> the save carries PARAM=3
#   7. -sigma 0:<s> alone           -> Suyama param0 (banner + param0-form save, no PARAM=)
#   8. -sigma 2:<s> alone           -> the save carries PARAM=2 (skipped if param2 is not
#                                      compiled into the binary, see -DECM_NO_PARAM2)
#   9. -sigma 1:<s>                 -> rejected as an unsupported parametrization
#
# usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools\test\test_cli_args.ps1
#          [-CudaExe build_cuda_cmake\ecm_cuda.exe] [-Device 1]
param(
    [string]$CudaExe = 'build_cuda_cmake\ecm_cuda.exe',
    [int]$Device = 1
)

$ErrorActionPreference = 'Continue'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not [System.IO.Path]::IsPathRooted($CudaExe)) { $CudaExe = Join-Path $repo $CudaExe }
if (-not (Test-Path $CudaExe)) { Write-Host "cuda exe not found: $CudaExe"; exit 2 }
$work = Join-Path $repo '.bench_tmp\test_cli_args'
New-Item -ItemType Directory -Force -Path $work | Out-Null
Get-ChildItem $work -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

$pass = 0; $fail = 0; $skip = 0
function Check([string]$name, [bool]$ok, [string]$detail = '') {
    if ($ok) { Write-Host "  [PASS] $name"; $script:pass++ }
    else { Write-Host "  [FAIL] $name"; if ($detail) { Write-Host "         $detail" }; $script:fail++ }
}
function Skip([string]$name, [string]$why) {
    Write-Host "  [SKIP] $name ($why)"; $script:skip++
}

# Small Mersenne prime (521) so a run is short and cannot find a factor.
[System.IO.File]::WriteAllText((Join-Path $work 'n.txt'), "(2^521-1)`n", ([System.Text.Encoding]::ASCII))

# Run the binary with the given argument tail; return @{exit=..; out=..}
function Run-Ecm([string]$argtail) {
    $argv = "-gpu -d $Device $argtail -gpucurves 1 --ckpt 0 1e3 0"
    $out = cmd /c "cd /d `"$work`" && `"$CudaExe`" $argv < n.txt" 2>&1 | Out-String
    return @{ exit = $LASTEXITCODE; out = $out }
}

# Run with -savea <file> and return the save's PARAM= field ('' when the param0 form has none)
function Run-Save([string]$argtail, [string]$saveName) {
    $save = Join-Path $work $saveName
    Remove-Item $save -Force -ErrorAction SilentlyContinue
    $r = Run-Ecm "$argtail -savea `"$save`""
    $param = ''
    if (Test-Path $save) {
        $first = Get-Content $save -TotalCount 1
        $m = [regex]::Match($first, 'PARAM=(\d+)')
        if ($m.Success) { $param = $m.Groups[1].Value }
    }
    return @{ exit = $r.exit; out = $r.out; param = $param; exists = (Test-Path $save) }
}

Write-Host "=== conflicting -sigma i:s vs --gpu-param ==="
foreach ($c in @(@{p = 0; s = 3}, @{p = 2; s = 3}, @{p = 3; s = 2})) {
    $r = Run-Ecm "--gpu-param $($c.p) -sigma $($c.s):12345678"
    $ok = ($r.exit -ne 0) -and ($r.out -match 'conflict between -sigma and -param arguments')
    Check "--gpu-param $($c.p) + -sigma $($c.s):s is rejected" $ok "exit=$($r.exit) out=$($r.out.Trim())"
}

Write-Host "=== agreeing / non-conflicting forms ==="
$r = Run-Ecm "--gpu-param 3 -sigma 3:12345678"
Check "--gpu-param 3 + -sigma 3:s is accepted" ($r.exit -eq 0) "exit=$($r.exit)"

$r = Run-Ecm "--gpu-param 0 -sigma 12345678"
Check "--gpu-param 0 + bare -sigma is accepted" ($r.exit -eq 0) "exit=$($r.exit)"

Write-Host "=== -sigma i:s alone selects the parametrization ==="
$r = Run-Save "-sigma 3:12345678" 'sel3.save'
Check "-sigma 3:s alone writes a PARAM=3 save" (($r.exit -eq 0) -and $r.param -eq '3') `
      "exit=$($r.exit) param='$($r.param)' exists=$($r.exists)"

$r = Run-Save "-sigma 0:12345678" 'sel0.save'
Check "-sigma 0:s alone selects Suyama param0" `
      (($r.exit -eq 0) -and ($r.out -match 'parametrization = Suyama param0') -and $r.param -eq '') `
      "exit=$($r.exit) param='$($r.param)'"

# param2 may be compiled out entirely (-DECM_NO_PARAM2=1); detect and report that instead of FAIL.
$probe = Run-Ecm "--gpu-param 2 -sigma 2:12345678"
if ($probe.out -match 'param2 kernels are NOT compiled into this binary') {
    Skip "-sigma 2:s alone writes a PARAM=2 save" "binary built with -DECM_NO_PARAM2=1"
} else {
    $r = Run-Save "-sigma 2:12345678" 'sel2.save'
    Check "-sigma 2:s alone writes a PARAM=2 save" (($r.exit -eq 0) -and $r.param -eq '2') `
          "exit=$($r.exit) param='$($r.param)'"
}

Write-Host "=== unsupported prefix ==="
$r = Run-Ecm "-sigma 1:12345678"
Check "-sigma 1:s is rejected as unsupported" (($r.exit -ne 0) -and ($r.out -match 'unsupported parametrization')) `
      "exit=$($r.exit)"

Write-Host ""
if ($fail -eq 0) { Write-Host "ALL OK ($pass checks$(if ($skip) { ", $skip skipped" }))" }
else { Write-Host "FAILED: $fail of $($pass + $fail)" }
exit $(if ($fail -eq 0) { 0 } else { 1 })
