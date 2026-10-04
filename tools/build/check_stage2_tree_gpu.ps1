#Requires -Version 5.1
<#
.SYNOPSIS
    Acceptance runner for the GPU tree stage-2 engine, slice S1 (+ the optional S2 tail).

.DESCRIPTION
    One command that produces the whole slice-1 evidence chain for a given (N, sigma, B1, B2, D):

      1. the CPU/GMP tree reference writes its F dump  (stage2_tree_ref --dump-F <file>);
      2. the GPU engine reads that dump and re-derives EVERYTHING from the Q inside it --
         the baby points with its own x-only Montgomery ladder, F with its own NTT product
         tree -- and compares point by point and coefficient by coefficient (mod N);
      3. it writes its own dump in the CPU's format, so the two files are compared line by
         line as well;
      4. with -Evaluate it then runs the stage-2 tail (giant points, remainder tree,
         accumulate, gcd) on the same machinery and its factor/hit_primes are compared with
         the CPU reference's summary line for the same shape.

    Deterministic: fixed N, sigma, bounds, D.  Device 1 by default (device 0 runs production
    stage 1).

.PARAMETER N
    Decimal modulus (default 2^128+1, the frozen vector).
.PARAMETER Sigma, B1, B2, D
    Curve/shape parameters (defaults: the frozen vector's 26 / 1000 / 1000000 / 210).
.PARAMETER Device
    CUDA device for the GPU engine (default 1).
.PARAMETER Evaluate
    Also run the stage-2 tail and compare the factor with the CPU reference.
.PARAMETER Sandbox
    Directory for the dump files (default tools\test\_run\stage2gpuc_<timestamp>).

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\check_stage2_tree_gpu.ps1 -Evaluate
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\check_stage2_tree_gpu.ps1 `
        -B1 1000 -B2 400000 -D 2310 -Evaluate
#>
param(
    [string]$N = '340282366920938463463374607431768211457',
    [int]$Sigma = 26,
    [int]$B1 = 1000,
    [int]$B2 = 1000000,
    [int]$D = 210,
    [int]$Device = 1,
    [switch]$Evaluate,
    # NTT_NAME_MAX for the GPU run: 0 (default) names EVERY hit leaf, so `hits`/`hit_primes`
    # are directly comparable with the CPU reference.  A positive value caps the DIAGNOSTIC
    # candidate scan (docs/DEV_STAGE2_GPU_PLAN.md section 27.3: at B2=1e11 a full scan would
    # take ~26 hours), which makes `hits`/`hit_primes` PARTIAL by construction -- the factor
    # set must still be identical, and that is what is asserted instead.
    [int]$NameMax = 0,
    # force the giant-point differential-addition chain and check it against the ladder
    [switch]$ChainCheck,
    # run the OLD host-side operand packing (NTT_S4_HOSTPACK=1) instead of the default device
    # packer, for the A/B: both must find the same factor set (section 33)
    [switch]$HostPack,
    [string]$Sandbox = ''
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo

$ref = Join-Path $repo 'build_cuda_cmake\stage2_tree_ref.exe'
$gpu = Join-Path $repo 'build_cuda_cmake\stage2_tree_gpu.exe'
foreach ($e in @($ref, $gpu)) {
    if (-not (Test-Path $e)) { throw "missing $e (build it: tools\build\build_stage2_tree_ref.ps1 / build_stage2_tree_gpu.ps1)" }
}
if (-not $Sandbox) { $Sandbox = Join-Path $repo ('tools\test\_run\stage2gpuc_' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Force $Sandbox | Out-Null
$cpu = Join-Path $Sandbox 'F_cpu.txt'
$gpuDump = Join-Path $Sandbox 'F_gpu.txt'

$tag = "N=$mid D=$D B1=$B1 B2=$B2 sigma=$Sigma"
Write-Host ("stage2 tree GPU check: {0}" -f $tag)
Write-Host ("sandbox: {0}" -f $Sandbox)

$refArgs = @('--n', $N, '--sigma', "$Sigma", '--b1', "$B1", '--b2', "$B2", '--d', "$D")
$cpuOut = (& $ref @refArgs 2>&1 | Out-String -Width 4096)
$cpuOutDump = (& $ref @refArgs --dump-F $cpu) 2>&1 | Out-String -Width 4096
if (-not (Test-Path $cpu)) { throw "the CPU reference did not write $cpu" }
$cpuSummary = [regex]::Match($cpuOutDump, 'stage2: algorithm=tree .*')
if (-not $cpuSummary.Success) { $cpuSummary = [regex]::Match($cpuOut, 'stage2: algorithm=tree .*') }
Write-Host ("  cpu: " + $cpuSummary.Value.Trim())

$gpuArgs = @('--check-F', $cpu, '--device', "$Device", '--dump-F-gpu', $gpuDump)
if ($Evaluate) { $gpuArgs += '--evaluate'; $gpuArgs += '--evaluate-batched' }
if ($NameMax -gt 0) { $env:NTT_NAME_MAX = "$NameMax" } else { Remove-Item Env:\NTT_NAME_MAX -ErrorAction SilentlyContinue }
# -ChainCheck: force the giant-point DIFFERENTIAL-ADDITION CHAIN (section 31) and make it compare
# itself against the old per-point ladder, point by point, on the affine x value.  Both give the
# same point in different projective representations, so X/Z mod N is the only comparable quantity
# -- and `mismatches=0` is the assertion that the chain is exact on this shape.
if ($ChainCheck) { $env:NTT_GIANT_CHAIN_MIN = '0'; $env:NTT_GIANT_CHAIN_CHECK = '1' }
else { Remove-Item Env:\NTT_GIANT_CHAIN_MIN -ErrorAction SilentlyContinue
       Remove-Item Env:\NTT_GIANT_CHAIN_CHECK -ErrorAction SilentlyContinue }
if ($HostPack) { $env:NTT_S4_HOSTPACK = '1' } else { Remove-Item Env:\NTT_S4_HOSTPACK -ErrorAction SilentlyContinue }
$gpuOut = (& $gpu @gpuArgs 2>&1 | Out-String -Width 4096)
$gpuOut.Trim() -split "`n" | ForEach-Object { Write-Host ("  gpu: " + $_.TrimEnd()) }
$gpuCode = $LASTEXITCODE

# --- the assertions -------------------------------------------------------------
$fail = 0
function Need([string]$what, $ok, [string]$detail = '') {
    if ($ok) { Write-Host ("  [ok]   " + $what) }
    else { Write-Host ("  [FAIL] " + $what + $(if ($detail) { " -- " + $detail } else { "" })) -ForegroundColor Red; $script:fail++ }
}
Need "the GPU engine exits 0" ($gpuCode -eq 0) ("exit=" + $gpuCode)
Need "device mod-N arithmetic matches GMP" ($gpuOut -match 'mont_selftest: cases=\d+ mismatches=0') ''
Need "every baby point x_j matches the CPU ladder" ($gpuOut -match 'ladder: baby_points=\d+ mismatches=0') ''
Need "every coefficient of F matches the CPU tree, mod N" ($gpuOut -match 'check_F: .*mismatches=0 .*baby_mismatches=0') ''
Need "F has the same degree as the CPU's" ($gpuOut -match 'check_F: F_degree_gpu=(\d+) F_degree_cpu=(\d+) ok=1') ''
Need "the exactness bound L*(2^bpw-1)^2 < p holds for the tree's own shapes" `
     ($gpuOut -match 'exactness: .* -> OK') ''
$diff = @(Compare-Object (Get-Content $cpu) (Get-Content $gpuDump) -SyncWindow 0)
Need "the two dump files are identical line by line" ($diff.Count -eq 0) ("differing=" + $diff.Count)

if ($ChainCheck) {
    # the chain must reproduce the ladder's points EXACTLY (affine x, point by point)
    Need "the giant chain matches the per-point ladder on every point" `
         ($gpuOut -match 'giant_chain_check: points=(\d+) blocks=\d+ per_block=\d+ seed_points=\d+ mismatches=0') `
         ($gpuOut -split "`n" | Where-Object { $_ -match 'giant_chain_check' } | Select-Object -First 1)
    Need "the giant chain was actually used (chunks >= 1)" `
         ($gpuOut -match '(real|batched)_giant_chain: chunks=[1-9]') ''
}

if ($Evaluate) {
    $g = [regex]::Match($gpuOut, 'stage2: algorithm=tree_gpu curves=(\d+) hits=(\d+) bad_factors=(\d+) factors=([^ ]*) hit_primes=([^ ]*)')
    $c = [regex]::Match($cpuSummary.Value, 'curves=(\d+) hits=(\d+) bad_factors=(\d+) factors=([^ ]*) hit_primes=([^ ]*)')
    Need "the GPU tail prints a summary line" ($g.Success) ''
    Need "the CPU reference prints a summary line" ($c.Success) ''
    if ($g.Success -and $c.Success) {
        Need "same factor set" ($g.Groups[4].Value -eq $c.Groups[4].Value) `
             ("gpu=" + $g.Groups[4].Value + " cpu=" + $c.Groups[4].Value)
        Need "same hit_primes" ($g.Groups[5].Value -eq $c.Groups[5].Value) `
             ("gpu=" + $g.Groups[5].Value + " cpu=" + $c.Groups[5].Value)
        Need "same hits" ($g.Groups[2].Value -eq $c.Groups[2].Value) `
             ("gpu=" + $g.Groups[2].Value + " cpu=" + $c.Groups[2].Value)
        Need "no bogus factor (bad_factors=0)" ($g.Groups[3].Value -eq '0') `
             ("bad_factors=" + $g.Groups[3].Value)
    }

    # ---- slice S3: the BATCHED structure must give the SAME answer as S2 and the CPU ----
    $b = [regex]::Match($gpuOut, 'stage2: algorithm=tree_gpu_batched curves=(\d+) hits=(\d+) bad_factors=(\d+) factors=([^ ]*) hit_primes=([^ ]*)')
    Need "the batched engine prints a summary line" ($b.Success) ''
    if ($b.Success) {
        if ($c.Success) {
            Need "batched: same factor set as the CPU reference" ($b.Groups[4].Value -eq $c.Groups[4].Value) `
                 ("batched=" + $b.Groups[4].Value + " cpu=" + $c.Groups[4].Value)
            if ($NameMax -gt 0) {
                # the naming scan is capped, so `hits`/`hit_primes` are PARTIAL by construction.
                # What must still hold: every prime the capped run DID name is one the CPU also
                # names (no bogus prime), and the factor set above is complete -- which is the
                # regression that the fallback record fixed (section 26.3).
                $cp = @($c.Groups[5].Value -split ',' | Where-Object { $_ })
                $gp = @($b.Groups[5].Value -split ',' | Where-Object { $_ })
                $subset = $true
                foreach ($p in $gp) { if ($cp -notcontains $p) { $subset = $false } }
                Need "batched: the capped run's hit_primes are a subset of the CPU's" $subset `
                     ("batched=" + $b.Groups[5].Value + " cpu=" + $c.Groups[5].Value)
                Need "batched: the cap was honoured (named_searches <= NTT_NAME_MAX)" `
                     ($gpuOut -match ('batched_naming: .*named_searches=(\d+) ') -and `
                      [int]$Matches[1] -le $NameMax) $gpuOut.Trim()
            } else {
                Need "batched: same hit_primes as the CPU reference" ($b.Groups[5].Value -eq $c.Groups[5].Value) `
                     ("batched=" + $b.Groups[5].Value + " cpu=" + $c.Groups[5].Value)
                Need "batched: same hits as the CPU reference" ($b.Groups[2].Value -eq $c.Groups[2].Value) `
                     ("batched=" + $b.Groups[2].Value + " cpu=" + $c.Groups[2].Value)
            }
        }
        if ($g.Success) {
            Need "batched: same factor set as the SIMPLE (S2) tail" ($b.Groups[4].Value -eq $g.Groups[4].Value) `
                 ("batched=" + $b.Groups[4].Value + " s2=" + $g.Groups[4].Value)
            Need "batched: same hit_primes as the SIMPLE (S2) tail" ($b.Groups[5].Value -eq $g.Groups[5].Value) `
                 ("batched=" + $b.Groups[5].Value + " s2=" + $g.Groups[5].Value)
        }
        Need "batched: no bogus factor (bad_factors=0)" ($b.Groups[3].Value -eq '0') `
             ("bad_factors=" + $b.Groups[3].Value)
        Need "batched: the poly_size is the CPU model's phi(D)/2" `
             ($gpuOut -match 'batched_shape: P=phi\(D\)/2=(\d+) giant_points=(\d+) num_poly_g=(\d+) loops=(\d+)') ''
        Need "batched: the folded structure is really the batched one (num_poly_g >= 1)" `
             ($gpuOut -match 'batched_shape: P=phi\(D\)/2=\d+ giant_points=\d+ num_poly_g=[1-9]') ''
        Need "batched: the multiplication counts and size distribution are reported" `
             ($gpuOut -match 'batched_cost: tree_convention=ours-padded .*poly_muls=\d+ operand_bits=\d+' -and `
              $gpuOut -match 'batched_cost_level: log2m=\d+ muls=\d+ operand_bits=\d+') ''
        Need "batched: the NTT call count is reported" ($gpuOut -match 'batched_ntt: ntt_calls_total=\d+') ''
        Need "batched: the descent's leaf values are accumulated ON THE DEVICE (block products)" `
             ($gpuOut -match 'prod_launches=[1-9]\d*') ''
        # the CPU cost model for the SAME batched structure, from the reference's own --cost
        $cm = [regex]::Match($cpuOutDump, 'cost_model: tree_convention=ours-padded S=\d+ P=phi\(D\)/2=(\d+) giant_points=(\d+) num_poly_g=(\d+)')
        Need "the CPU reference prints its batched cost model (ours-padded)" ($cm.Success) ''
        if ($cm.Success -and ($gpuOut -match 'batched_shape: P=phi\(D\)/2=(\d+) giant_points=(\d+) num_poly_g=(\d+)')) {
            $bs = [regex]::Match($gpuOut, 'batched_shape: P=phi\(D\)/2=(\d+) giant_points=(\d+) num_poly_g=(\d+)')
            Need "batched: same poly_size P as the CPU model" ($bs.Groups[1].Value -eq $cm.Groups[1].Value) `
                 ("gpu=" + $bs.Groups[1].Value + " cpu=" + $cm.Groups[1].Value)
            Need "batched: same giant point count as the CPU model" ($bs.Groups[2].Value -eq $cm.Groups[2].Value) `
                 ("gpu=" + $bs.Groups[2].Value + " cpu=" + $cm.Groups[2].Value)
        }
    }
}

Write-Host ""
if ($fail -gt 0) { Write-Host ("FAILED: " + $fail) -ForegroundColor Red; exit 1 }
Write-Host "all checks passed"
exit 0
