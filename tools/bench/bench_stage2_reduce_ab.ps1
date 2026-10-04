#Requires -Version 5.1
<#
.SYNOPSIS
    Same-binary ABBA benchmark of reduction, sampling, packing, chunk budget or scaled descent.
.DESCRIPTION
    Defaults to the production shape in DEV_GPUOWL_NTT_NOTES.md section 32.  Writes a log
    for each run, provenance.json and results.csv.  The mode, GMP checks, factors/hit primes,
    coefficient count and arena overflow are checked before a speed comparison is reported.
    All environment overrides are restored.  Device 1 is used unless explicitly overridden.
#>
param(
    [string]$Exe = '',
    [string]$NHex = ('1' + ('f' * 1315)),
    [UInt64]$B1 = 1000,
    [UInt64]$B2 = 1940000000000,
    [UInt64]$D = 1231230,
    [int]$Sigma = 26,
    [ValidateSet(1,12)][int]$Stage1Extra = 1,
    [string]$ExpectedQHex = '',
    [int]$Device = 1,
    [ValidateSet('reduction','oracle','oracle_pack','carry_batch','pack_direct','batch_mb','flat_direct','groot','workspace','fuse_scratch','final_readback','output_window','chunk_output','scaled_descent','groot_device','groot_memory','groot_carry','gfinv_batch','fold_flat','seed_device','mersenne','small_prime')][string]$Target = 'reduction',
    [ValidateRange(1,256)][int]$BatchMB = 32,
    [ValidateRange(1,256)][int]$CandidateBatchMB = 64,
    [ValidateRange(0,65536)][int]$ArenaMB = 0,
    [ValidateRange(0,2147483647)][int]$ChainMin = 32768,
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
if (-not $Exe) { $Exe = Join-Path $repo 'build_cuda_cmake\stage2_tree_gpu.exe' }
if (-not (Test-Path -LiteralPath $Exe)) { throw "missing binary: $Exe" }
if ($Target -eq 'batch_mb' -and $BatchMB -eq $CandidateBatchMB) { throw 'chunk budgets must differ for A/B' }
if (-not $Output) { $Output = Join-Path $repo ('build_cuda_cmake\_reduce_ab_' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$binaryHash = (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash
$runArgs = @('--real', '--n-hex', $NHex, '--sigma', "$Sigma", '--b1', "$B1", '--b2', "$B2",
             '--d', "$D", '--device', "$Device")
$order = @('montgomery', 'division', 'division', 'montgomery')
if ($Target -eq 'oracle') { $order = @('blocking', 'oracle_async', 'oracle_async', 'blocking') }
if ($Target -eq 'oracle_pack') { $order = @('gmp_digits', 'limb_pack', 'limb_pack', 'gmp_digits') }
if ($Target -in @('carry_batch','groot_carry')) { $order = @('carry_per_chunk', 'carry_batch', 'carry_batch', 'carry_per_chunk') }
if ($Target -eq 'pack_direct') { $order = @('pack_copy', 'pack_direct', 'pack_direct', 'pack_copy') }
if ($Target -eq 'batch_mb') { $order = @("batch_$BatchMB", "batch_$CandidateBatchMB", "batch_$CandidateBatchMB", "batch_$BatchMB") }
if ($Target -eq 'flat_direct') { $order = @('flat_copy','flat_direct','flat_direct','flat_copy') }
if ($Target -eq 'groot') { $order = @('full_gtree','groot','groot','full_gtree') }
if ($Target -eq 'workspace') { $order = @('keyed_workspace','workspace_pool','workspace_pool','keyed_workspace') }
if ($Target -eq 'fuse_scratch') { $order = @('wide_scratch','compact_scratch','compact_scratch','wide_scratch') }
if ($Target -eq 'final_readback') { $order = @('whole_readback','chunk_readback','chunk_readback','whole_readback') }
if ($Target -eq 'output_window') { $order = @('full_output','output_window','output_window','full_output') }
if ($Target -eq 'chunk_output') { $order = @('whole_output_buffer','chunk_output_buffer','chunk_output_buffer','whole_output_buffer') }
if ($Target -eq 'scaled_descent') { $order = @('division_descent','scaled_descent','scaled_descent','division_descent') }
if ($Target -eq 'groot_device') { $order = @('host_groot','device_groot','device_groot','host_groot') }
if ($Target -eq 'groot_memory') { $order = @('legacy_gmemory','compact_gmemory','compact_gmemory','legacy_gmemory') }
if ($Target -eq 'small_prime') { $order = @('small_ladder','baby_reuse','baby_reuse','small_ladder') }
if ($Target -eq 'mersenne') { $order = @('division','mersenne','mersenne','division') }
if ($Target -eq 'seed_device') { $order = @('host_seed','device_seed','device_seed','host_seed') }
if ($Target -eq 'fold_flat') { $order = @('vector_fold','flat_fold','flat_fold','vector_fold') }
if ($Target -eq 'gfinv_batch') { $order = @('segment_individual','segment_batch','segment_batch','segment_individual') }
$overrides = @{ NTT_NAME_MAX='1'; NTT_S4_BATCH_MB="$BatchMB"; NTT_S4_ASYNC='1';
                NTT_S4_DEFER_CARRY='1'; NTT_S4_HOSTPACK='0'; NTT_S5_ON='0'; NTT_S4_OLDTAIL='1';
                NTT_S5_REDDUMP='0'; NTT_S4_ORACLE_ASYNC='0'; NTT_S4_ORACLE_RING='4'; NTT_S4_ORACLE_PACK='1';
                NTT_S4_ORACLE_TEST_BAD='0'; NTT_S4_SAMPLE='96'; NTT_S4_CHECK_EVERY='8';
                NTT_S4_CARRY_BATCH='0'; NTT_S4_CHUNK_MAX='0'; NTT_S4_CARRY_TEST_BAD='0';
                NTT_S4_CARRY_TRACE='0'; NTT_S4_PACK_DIRECT='1'; NTT_S4_FLAT_DIRECT='1'; NTT_S4_FLAT_TEST='0';
                NTT_CARRY_ROUNDS=''; NTT_S4_GROOT_ONLY='1'; NTT_S4_GROOT_TEST='0'; NTT_S4_OFF='0';
                NTT_ARENA_WORKSPACE_POOL='1'; NTT_ARENA_WORKSPACE_TEST='0';
                NTT_FUSE_COMPACT_SCRATCH='1'; NTT_FUSE_LIFETIME_TEST='0';
                NTT_S4_FINAL_READBACK='0'; NTT_S4_FINAL_READBACK_TEST='0';
                NTT_S4_OUTPUT_WINDOW='0'; NTT_S4_OUTPUT_WINDOW_TEST='0'; NTT_S4_CHUNK_OUTPUT='0';
                NTT_GROOT_LEAF_STAGING='1'; NTT_GROOT_COMPACT_RAW='1'; NTT_GROOT_LEAF_CHUNK='0';
                NTT_GROOT_DEVICE='0'; NTT_GROOT_DEVICE_TEST='0'; NTT_GROOT_DEVICE_CHECK='0'; NTT_GROOT_DEVICE_TEST_BAD='0';
                NTT_SCALED_DESCENT='0'; NTT_SCALED_TEST='0'; NTT_SCALED_CHECK='0'; NTT_S4_DESCENT_CHECK='0';
                NTT_SMALL_PRIME_REUSE='0'; NTT_SMALL_PRIME_CHECK='0'; NTT_SMALL_PRIME_TEST_BAD='0'; NTT_SMALL_PRIME_CACHE_STALE='0';
                NTT_S4_MERSENNE='0'; NTT_S4_MERSENNE_TEST='0'; NTT_S4_MERSENNE_TEST_BAD='0';
                NTT_GIANT_SEED_DEVICE='0'; NTT_GIANT_SEED_CHECK='0'; NTT_GFINV_SEG_EXACT='1'; NTT_GFINV_SEG_CHECK='0';
                NTT_GFINV_SEG_TEST='0'; NTT_GFINV_SEG_TEST_BAD='0';
                NTT_FOLD_FLAT='0'; NTT_FOLD_FLAT_TEST='0'; NTT_FOLD_FLAT_TEST_BAD='0';
                NTT_GFINV_BATCH='0'; NTT_GFINV_BATCH_TEST='0'; NTT_GFINV_BATCH_TEST_BAD='0'; NTT_REAL_F_DUMP='';
                NTT_STAGE1_EXTRA="$Stage1Extra"; NTT_STAGE1_Q_DUMP=$(if($ExpectedQHex -or $Stage1Extra -eq 12){'1'}else{'0'}) }
if ($ArenaMB -gt 0) { $overrides.NTT_ARENA_CAP_KB = "$([long]$ArenaMB * 1024)" }
if ($Target -in @('seed_device','mersenne','small_prime')) {
    $overrides.NTT_GIANT_CHAIN_MIN="$ChainMin"
    $overrides.NTT_GIANT_CHAIN_BLOCK='64'
    $overrides.NTT_GIANT_LADDER='0'
    $overrides.NTT_GIANT_CHAIN_CHECK='0'
}
if($Target -eq 'mersenne' -and ($order -join ',') -ne 'division,mersenne,mersenne,division'){throw 'Mersenne ABBA order invalid'}
if($Target -eq 'small_prime' -and ($order -join ',') -ne 'small_ladder,baby_reuse,baby_reuse,small_ladder'){throw 'Small-prime ABBA order invalid'}
$saved = @{}
foreach ($key in $overrides.Keys) { $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
if($Target -eq 'fold_flat' -and ($order -join ',') -ne 'vector_fold,flat_fold,flat_fold,vector_fold'){throw 'flat fold ABBA order invalid'}
if($Target -eq 'seed_device' -and ($order -join ',') -ne 'host_seed,device_seed,device_seed,host_seed'){throw 'seed ABBA order invalid'}
$modeControls = @(foreach ($mode in $order) {
    [pscustomobject]@{ mode=$mode;
        NTT_SMALL_PRIME_REUSE=$(if($mode -eq 'baby_reuse'){'1'}else{'0'});
        NTT_S4_MERSENNE=$(if($mode -eq 'mersenne' -or $Target -eq 'small_prime'){'1'}else{'0'});
        NTT_GIANT_SEED_DEVICE=$(if ($mode -eq 'device_seed' -or $Target -in @('mersenne','small_prime')) {'1'}else{'0'});
        NTT_GFINV_SEG_EXACT='1';
        NTT_FOLD_FLAT=$(if ($mode -eq 'flat_fold' -or $Target -in @('seed_device','mersenne','small_prime')) { '1' } else { '0' });
        NTT_GFINV_BATCH=$(if ($mode -eq 'segment_batch' -or $Target -in @('fold_flat','seed_device','mersenne','small_prime')) { '1' } else { '0' });
        NTT_S4_OLDTAIL=$(if ($mode -eq 'montgomery') { '1' } else { '0' });
        NTT_S4_ORACLE_ASYNC=$(if ($mode -eq 'oracle_async') { '1' } else { '0' });
        NTT_S4_ORACLE_PACK=$(if ($mode -eq 'gmp_digits') { '0' } else { '1' });
        NTT_S4_CARRY_BATCH=$(if ($mode -eq 'carry_batch') { '1' } else { '0' });
        NTT_S4_PACK_DIRECT=$(if ($Target -eq 'pack_direct' -and $mode -eq 'pack_copy') { '0' } else { '1' });
        NTT_S4_FLAT_DIRECT=$(if ($mode -eq 'flat_copy') { '0' } else { '1' });
        NTT_S4_GROOT_ONLY=$(if ($mode -eq 'full_gtree') { '0' } else { '1' });
        NTT_ARENA_WORKSPACE_POOL=$(if ($mode -eq 'keyed_workspace') { '0' } else { '1' });
        NTT_FUSE_COMPACT_SCRATCH=$(if ($mode -eq 'wide_scratch') { '0' } else { '1' });
        NTT_S4_FINAL_READBACK=$(if ($mode -eq 'whole_readback') { '1' } else { '0' });
        NTT_S4_OUTPUT_WINDOW=$(if ($mode -eq 'output_window' -or $Target -in @('scaled_descent','groot_device','groot_memory','groot_carry','gfinv_batch','fold_flat','seed_device','mersenne','small_prime')) { '1' } else { '0' });
        NTT_S4_CHUNK_OUTPUT=$(if ($mode -eq 'chunk_output_buffer' -or $Target -in @('scaled_descent','groot_device','groot_memory','groot_carry','gfinv_batch','fold_flat','seed_device','mersenne','small_prime')) { '1' } else { '0' });
        NTT_GROOT_DEVICE=$(if ($mode -eq 'device_groot' -or $Target -in @('groot_memory','groot_carry','gfinv_batch','fold_flat','seed_device','mersenne','small_prime')) { '1' } else { '0' });
        NTT_GROOT_LEAF_STAGING=$(if ($mode -eq 'legacy_gmemory') { '0' } else { '1' });
        NTT_GROOT_COMPACT_RAW=$(if ($mode -eq 'legacy_gmemory') { '0' } else { '1' });
        NTT_SCALED_DESCENT=$(if ($mode -eq 'scaled_descent' -or $Target -in @('groot_device','groot_memory','groot_carry','gfinv_batch','fold_flat','seed_device','mersenne','small_prime')) { '1' } else { '0' });
        NTT_S4_BATCH_MB=$(if ($Target -eq 'batch_mb' -and $mode -eq "batch_$CandidateBatchMB") { "$CandidateBatchMB" } else { "$BatchMB" }) }
})
@{ exe=$Exe; sha256=$binaryHash; args=$runArgs; order=$order; target=$Target; env=$overrides;
   mode_controls=$modeControls; started=(Get-Date -Format o); head=(& git rev-parse HEAD) } |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Output 'provenance.json') -Encoding UTF8
$rows = @()
try {
    foreach ($key in $overrides.Keys) { [Environment]::SetEnvironmentVariable($key, $overrides[$key], 'Process') }
    for ($i = 0; $i -lt $order.Count; ++$i) {
        $mode = $order[$i]
        $algorithm = $(if ($mode -eq 'montgomery') { 'montgomery' } elseif($mode -eq 'mersenne' -or $Target -eq 'small_prime'){'mersenne'} else { 'division' })
        $env:NTT_SMALL_PRIME_REUSE=$(if($mode -eq 'baby_reuse'){'1'}else{'0'})
        $env:NTT_S4_MERSENNE=$(if($mode -eq 'mersenne' -or $Target -eq 'small_prime'){'1'}else{'0'})
        $oracleAsync = $(if ($mode -eq 'oracle_async') { '1' } else { '0' })
        $oraclePack = $(if ($mode -eq 'gmp_digits') { '0' } else { '1' })
        $carryBatch = $(if ($mode -eq 'carry_batch') { '1' } else { '0' })
        $packDirect = $(if ($Target -eq 'pack_direct' -and $mode -eq 'pack_copy') { '0' } else { '1' })
        $batchBudget = $(if ($Target -eq 'batch_mb' -and $mode -eq "batch_$CandidateBatchMB") { $CandidateBatchMB } else { $BatchMB })
        $flatDirect = $(if ($mode -eq 'flat_copy') { '0' } else { '1' })
        $grootOnly = $(if ($mode -eq 'full_gtree') { '0' } else { '1' })
        $env:NTT_FOLD_FLAT = $(if ($mode -eq 'flat_fold' -or $Target -in @('seed_device','mersenne','small_prime')) { '1' } else { '0' })
        $env:NTT_GIANT_SEED_DEVICE = $(if ($mode -eq 'device_seed' -or $Target -in @('mersenne','small_prime')) {'1'}else{'0'})
        $env:NTT_GFINV_BATCH = $(if ($mode -eq 'segment_batch' -or $Target -in @('fold_flat','seed_device','mersenne','small_prime')) { '1' } else { '0' })
        $env:NTT_S4_OLDTAIL = $(if ($algorithm -eq 'montgomery') { '1' } else { '0' })
        $env:NTT_S4_ORACLE_ASYNC = $oracleAsync
        $env:NTT_S4_ORACLE_PACK = $oraclePack
        $env:NTT_S4_CARRY_BATCH = $carryBatch
        $env:NTT_S4_PACK_DIRECT = $packDirect
        $env:NTT_S4_BATCH_MB = "$batchBudget"
        $env:NTT_S4_FLAT_DIRECT = $flatDirect
        $env:NTT_S4_GROOT_ONLY = $grootOnly
        $env:NTT_ARENA_WORKSPACE_POOL = $(if ($mode -eq 'keyed_workspace') { '0' } else { '1' })
        $env:NTT_FUSE_COMPACT_SCRATCH = $(if ($mode -eq 'wide_scratch') { '0' } else { '1' })
        $env:NTT_S4_FINAL_READBACK = $(if ($mode -eq 'whole_readback') { '1' } else { '0' })
        $env:NTT_S4_OUTPUT_WINDOW = $(if ($mode -eq 'output_window' -or $Target -in @('scaled_descent','groot_device','groot_memory','groot_carry','gfinv_batch','fold_flat','seed_device','mersenne','small_prime')) { '1' } else { '0' })
        $env:NTT_S4_CHUNK_OUTPUT = $(if ($mode -eq 'chunk_output_buffer' -or $Target -in @('scaled_descent','groot_device','groot_memory','groot_carry','gfinv_batch','fold_flat','seed_device','mersenne','small_prime')) { '1' } else { '0' })
        $env:NTT_GROOT_DEVICE = $(if ($mode -eq 'device_groot' -or $Target -in @('groot_memory','groot_carry','gfinv_batch','fold_flat','seed_device','mersenne','small_prime')) { '1' } else { '0' })
        $env:NTT_SCALED_DESCENT = $(if ($mode -eq 'scaled_descent' -or $Target -in @('groot_device','groot_memory','groot_carry','gfinv_batch','fold_flat','seed_device','mersenne','small_prime')) { '1' } else { '0' })
        $env:NTT_GROOT_LEAF_STAGING = $(if ($mode -eq 'legacy_gmemory') { '0' } else { '1' })
        $env:NTT_GROOT_COMPACT_RAW = $(if ($mode -eq 'legacy_gmemory') { '0' } else { '1' })
        $residentMode = $env:NTT_GROOT_DEVICE -eq '1'
        if ((Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash -ne $binaryHash) {
            throw 'binary changed during A/B; comparison invalid'
        }
        $log = Join-Path $Output (('{0}_{1}.log' -f ($i+1), $mode))
        Write-Host ("run {0}/4: {1}; log={2}" -f ($i+1), $mode, $log)
        $runStarted = Get-Date
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $ErrorActionPreference = 'Continue'   # capture native stderr in the evidence log
        & $Exe @runArgs > $log 2>&1
        $rc = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        $sw.Stop()
        $runEnded = Get-Date
        $text = Get-Content -LiteralPath $log -Raw
        $qHex=[regex]::Match($text,'real_setup_Q_full: hex=([0-9a-f]+)').Groups[1].Value
        if($text -notmatch "stage1_extra=$Stage1Extra(?:\s|$)" -or
           ($ExpectedQHex -and $qHex -cne $ExpectedQHex.ToLowerInvariant()) -or
           ($env:NTT_STAGE1_Q_DUMP -eq '1' -and -not $qHex)) {
            throw "Stage1 Q/scalar contract failed; inspect $log"
        }
        $stage = [regex]::Match($text, 'stage2:.*?hits=(\d+) bad_factors=(\d+) factors=([^\s]*) hit_primes=([^\s]*) elapsed=([0-9.]+)')
        $leaves = [regex]::Match($text, 'descent_values: leaves=(\d+) words=(\d+) hash=(\d+)')
        $shape = [regex]::Match($text, 'real_batched_shape: P=(\d+) giant_points=(\d+) num_poly_g=(\d+) loops=(\d+) descent_divmods=(\d+)')
        $scaledLine = [regex]::Match($text, '(?m)^scaled_descent:.*').Value
        $scaled = @{}
        foreach($field in @('enabled','levels','mul_calls','mul_pairs','copies','zeros','states','words','leaves',
                            'checked_states','checked_words','frontier_peak_bytes','pack_peak_bytes',
                            'root_inverse_reused','root_divisions')) {
            $m=[regex]::Match($scaledLine,"(?:^| )$field=(\d+)")
            if($env:NTT_SCALED_DESCENT -eq '1' -and -not $m.Success){throw "missing scaled counter: $field; inspect $log"}
            $scaled[$field]=$(if($m.Success){[UInt64]$m.Groups[1].Value}else{[UInt64]0})
        }
        if($Target -eq 'scaled_descent' -and (-not $leaves.Success -or -not $shape.Success -or
           ($mode -eq 'scaled_descent' -and ($scaled.enabled -ne 1 -or $scaled.mul_calls -le 0 -or
             $scaled.leaves -ne [UInt64]$leaves.Groups[1].Value -or $scaled.checked_states -ne 0 -or $scaled.checked_words -ne 0 -or
             $shape.Groups[5].Value -ne '0' -or
             ($shape.Groups[4].Value -ne '0' -and $scaled.root_inverse_reused -ne 1))) -or
           ($mode -ne 'scaled_descent' -and $scaledLine))) {
            throw "scaled descent path/output contract failed; inspect $log"
        }
        $ginvLine=[regex]::Match($text,'(?m)^real_batched_gfinv:.*').Value;$ginv=@{}
        foreach($field in @('enabled','requests','cache_hits','groups','segments','group_attempts','group_failures',
                            'individual_attempts','good','nonunits','scratch_peak_bytes','t_prepare')) {
            $m=[regex]::Match($ginvLine,"(?:^| )$field=([0-9.]+)")
            if($m.Success){$ginv[$field]=[double]$m.Groups[1].Value}
        }
        if($Target -eq 'gfinv_batch' -and ($ginv.Count -ne 12 -or "$($ginv.enabled)" -ne $env:NTT_GFINV_BATCH -or
           $ginv.requests -le 0 -or
           ($ginv.enabled -eq 0 -and ($ginv.groups -ne 0 -or $ginv.scratch_peak_bytes -ne 0 -or $ginv.individual_attempts -ne $ginv.requests)) -or
           ($ginv.enabled -eq 1 -and ($ginv.groups -le 0 -or $ginv.group_attempts -ne $ginv.groups -or
            $ginv.good+$ginv.nonunits -ne $ginv.segments -or $ginv.scratch_peak_bytes -le 0 -or
            $ginv.group_attempts+$ginv.individual_attempts -gt $ginv.segments+$ginv.groups)))) {
            throw "Segment inverse control/accounting failed; inspect $log"
        }
        $foldLine=[regex]::Match($text,'(?m)^real_batched_foldflat:.*').Value;$foldFlat=@{}
        foreach($field in @('enabled','folds','muls','sub_coeffs','peak_bytes','prepare','multiply','subtract','bridge')) {
            $m=[regex]::Match($foldLine,"(?:^| )$field=([0-9.]+)")
            if($m.Success){$foldFlat[$field]=[double]$m.Groups[1].Value}
        }
        if($Target -eq 'fold_flat' -and ($foldFlat.Count -ne 9 -or "$($foldFlat.enabled)" -ne $(if($mode -eq 'flat_fold' -or $Target -in @('seed_device','mersenne','small_prime')){'1'}else{'0'}) -or
           $ginv.enabled -ne 1 -or $shape.Groups[4].Value -eq '0' -or
           ($foldFlat.enabled -eq 1 -and ($foldFlat.folds -ne [UInt64]$shape.Groups[4].Value -or
             $foldFlat.muls -le 0 -or $foldFlat.sub_coeffs -le 0 -or $foldFlat.peak_bytes -le 0 -or
             $scaled.root_inverse_reused -ne 1)))) {throw "flat fold accounting/inverse reuse failed; inspect $log"}
        $gleaf=[regex]::Match($text,'real_batched_gleaves: in=([0-9.]+) invert=([0-9.]+) out=([0-9.]+).*gscale=([0-9.]+)')
        $projective=[regex]::Match($text,'real_batched_projective: leaves=(\d+) gamma_points=(\d+) segments=(\d+) affine_fallback_points=(\d+)')
        $gdeviceLine=[regex]::Match($text,'(?m)^real_batched_gdevice:.*').Value;$gdevice=@{}
        foreach($field in @('enabled','trees','fallbacks','levels','groups','pairs','copies','leaf_words','root_words',
                            'resident_words','trace_words','metadata_words','metadata_peak_bytes','raw_peak_bytes',
                            'logical_frontier_peak_bytes','host_staging_peak_bytes','checked_nodes','checked_words')) {
            $m=[regex]::Match($gdeviceLine,"(?:^| )$field=(\d+)")
            if($m.Success){$gdevice[$field]=[UInt64]$m.Groups[1].Value}
        }
        if($Target -in @('groot_device','groot_memory','groot_carry','gfinv_batch','fold_flat','seed_device','mersenne','small_prime') -and ($gdevice.Count -ne 18 -or
           "$($gdevice.enabled)" -ne $env:NTT_GROOT_DEVICE -or $scaled.enabled -ne 1 -or
           $scaled.checked_states -ne 0 -or $scaled.checked_words -ne 0 -or
           ($residentMode -and ($gdevice.trees -le 0 -or $gdevice.fallbacks -ne 0 -or
               $gdevice.resident_words -le 0 -or $gdevice.trace_words -ne 0 -or
               $gdevice.metadata_words -ne 3*$gdevice.pairs -or $gdevice.checked_nodes -ne 0 -or $gdevice.checked_words -ne 0)) -or
           (-not $residentMode -and ($gdevice.trees -ne 0 -or $gdevice.resident_words -ne 0)))) {
            throw "resident G-root path/accounting failed; inspect $log"
        }
        $mersenne=[regex]::Match($text,'s4_mersenne_mode: requested=(\d+) eligible=(\d+) bits=(\d+) enabled=(\d+)')
        if(-not $mersenne.Success -or $mersenne.Groups[1].Value -ne $env:NTT_S4_MERSENNE -or
           $mersenne.Groups[4].Value -ne $env:NTT_S4_MERSENNE -or
           ($Target -eq 'mersenne' -and $mersenne.Groups[2].Value -ne '1')) {
            throw "Mersenne reduction selector failed; inspect $log"
        }
        $small=@{}
        if($Target -eq 'small_prime') {
            $line=[regex]::Match($text,'(?m)^small_prime_reuse:.*').Value
            foreach($field in @('requested','available','matched','primes','reused','fallback','checked','bad','avoided_montmuls','avoided_h2d_bytes','avoided_d2h_bytes','cache_bytes','elapsed')) {
                $m=[regex]::Match($line,"(?:^| )$field=([0-9.]+)")
                if($m.Success){$small[$field]=[double]$m.Groups[1].Value}
            }
            $enabled=[int]$env:NTT_SMALL_PRIME_REUSE
            $w=[int][Math]::Ceiling(($NHex.TrimStart('0').Length*4)/64.0)
            if($small.Count -ne 13 -or $small.requested -ne $enabled -or $small.matched -ne $enabled -or
               $small.available -ne $enabled -or $small.checked -ne 0 -or $small.bad -ne 0 -or
               $small.primes -ne $small.reused+$small.fallback -or
               $small.avoided_h2d_bytes -ne 8*$small.reused -or $small.avoided_d2h_bytes -ne 16*$w*$small.reused -or
               ($enabled -eq 1 -and ($small.reused -le 0 -or $small.cache_bytes -le 0 -or $small.avoided_montmuls -le 0)) -or
               ($enabled -eq 0 -and ($small.reused -ne 0 -or $small.cache_bytes -ne 0))) {
                throw "Small-prime reuse control/accounting failed; inspect $log"
            }
        }
        $seed=@{}
        if($Target -in @('seed_device','mersenne','small_prime')) {
            $line=[regex]::Match($text,'(?m)^real_giant_seed:.*').Value
            foreach($field in @('enabled','exact_segments','chunks','points','avoided_d2h_bytes','avoided_h2d_bytes',
                'avoided_cpu_modmuls','avoided_montmuls','checked_words','segments','segment_checks','segment_fix_muls','fix_table_peak_bytes')) {
                $m=[regex]::Match($line,"(?:^| )$field=(\d+)")
                if($m.Success){$seed[$field]=[UInt64]$m.Groups[1].Value}
            }
            $expected=$(if($mode -eq 'device_seed' -or $Target -in @('mersenne','small_prime')){'1'}else{'0'})
            $w=[int][Math]::Ceiling(($NHex.TrimStart('0').Length*4)/64.0)
            if($seed.Count -ne 13 -or "$($seed.enabled)" -ne $expected -or $seed.exact_segments -ne 1 -or
               $seed.checked_words -ne 0 -or $seed.segment_checks -ne 0 -or $seed.fix_table_peak_bytes -le 0 -or
               ($expected -eq '1' -and ($seed.chunks -le 0 -or $seed.points -le 0 -or
                 $seed.avoided_d2h_bytes -ne 16*$w*$seed.points -or $seed.avoided_h2d_bytes -ne $seed.avoided_d2h_bytes -or
                 $seed.avoided_cpu_modmuls -ne 2*$seed.points -or $seed.avoided_montmuls -ne 2*$seed.points)) -or
               ($expected -eq '0' -and ($seed.points -ne 0 -or $seed.avoided_d2h_bytes -ne 0 -or $seed.avoided_h2d_bytes -ne 0))) {
                throw "Device seed path/accounting failed; inspect $log"
            }
        }
        $gmemoryLine=[regex]::Match($text,'(?m)^real_batched_gmemory:.*').Value;$gmemory=@{}
        foreach($field in @('compact_raw','leaf_staging','pinned_trees','pinned_slices','pinned_words','pageable_words',
                            'leaf_fallbacks','legacy_trees','pinned_borrow_peak_bytes','rawA_peak_bytes','rawB_peak_bytes')) {
            $m=[regex]::Match($gmemoryLine,"(?:^| )$field=(\d+)")
            if($m.Success){$gmemory[$field]=[UInt64]$m.Groups[1].Value}
        }
        if($Target -in @('groot_memory','groot_carry','gfinv_batch','fold_flat','seed_device','mersenne','small_prime') -and ($gmemory.Count -ne 11 -or
           "$($gmemory.compact_raw)" -ne $env:NTT_GROOT_COMPACT_RAW -or "$($gmemory.leaf_staging)" -ne $env:NTT_GROOT_LEAF_STAGING -or
           $gmemory.pinned_words+$gmemory.pageable_words -ne $gdevice.leaf_words -or
           $gmemory.rawA_peak_bytes+$gmemory.rawB_peak_bytes -ne $gdevice.raw_peak_bytes -or
           ($env:NTT_GROOT_LEAF_STAGING -eq '1' -and ($gmemory.pinned_trees -ne $gdevice.trees -or $gmemory.leaf_fallbacks -ne 0 -or
             $gmemory.legacy_trees -ne 0 -or $gmemory.pageable_words -ne 0 -or $gdevice.host_staging_peak_bytes -ne 0)) -or
           ($mode -eq 'legacy_gmemory' -and ($gmemory.legacy_trees -ne $gdevice.trees -or $gmemory.pinned_words -ne 0)))) {
            throw "G-root memory path/accounting failed; inspect $log"
        }
        $reduce = [regex]::Match($text, 's4_multiply_stats:.*?coeffs_reduced=(\d+) t_reduce=([0-9.]+)')
        $arena = [regex]::Match($text, 'real_batched_breakdown:.*?arena_overflow=(\d+)')
        $carry = [regex]::Match($text, 'real_batched_carrydefer: chunks_deferred=(\d+) finishes=(\d+) deferred_slices=(\d+) batch_enabled=(\d+) checked_chunks=(\d+) max_group=(\d+)')
        $upload = [regex]::Match($text, 'raw_reuse_waits=(\d+) raw_reuse_wait=([0-9.]+) raw_pinned_bytes=(\d+)')
        $transfers = [regex]::Match($text, 'real_batched_asyncxfer: raw_async=(\d+) out_async=(\d+) fallbacks=0 \(async_enabled=1\)')
        $rawVolume = [regex]::Match($text, 'real_batched_rawupload:.*?total=([0-9.]+) s volume=([0-9.]+) GB.*?pack_launches=(\d+)')
        $backVolume = [regex]::Match($text, 'real_batched_coeffback:.*?total=([0-9.]+) s volume=([0-9.]+) GB')
        $carryTime = [regex]::Match($text, 'real_batched_carrysplit:.*?d2h=([0-9.]+) s')
        $carryLedger = [regex]::Match($text, 'real_batched_carrytime: group_readback=([0-9.]+) chunk_readback=([0-9.]+) total=([0-9.]+)')
        $inputPack = [regex]::Match($text, 'real_batched_input: direct_enabled=(\d+) direct_chunks=(\d+) copied_chunks=(\d+) d2d_bytes=(\d+) avoided_bytes=(\d+) packed_peak_bytes=(\d+) temp_peak_bytes=(\d+) temp_current_bytes=(\d+) pack_host=([0-9.]+) copy_host=([0-9.]+)')
        $phaseSplit = [regex]::Match($text, 'real_batched_split: giant=([0-9.]+) gtrees=([0-9.]+) fold=([0-9.]+) descent=([0-9.]+) inv=([0-9.]+) accum=([0-9.]+) name=([0-9.]+) f_tree_incl=([0-9.]+)')
        $flat = [regex]::Match($text, 'real_batched_flatinput: direct_enabled=(\d+) calls=(\d+) borrowed=(\d+) padded=(\d+) alias_clones=(\d+) copy_bytes=(\d+) zero_bytes=(\d+) avoided_copy_bytes=(\d+) avoided_zero_bytes=(\d+) temp_peak_bytes=(\d+) control_peak_bytes=(\d+) t_prepare=([0-9.]+)')
        $groot = [regex]::Match($text, 'real_batched_groot: root_only=(\d+) builds=(\d+) nodes_released=(\d+) moves=(\d+) node_peak_bytes=(\d+) retained_peak_bytes=(\d+) released_bytes=(\d+) input_released_bytes=(\d+) root_words=(\d+) trace=(\d+) root_hash=([0-9a-f]+) t_release=([0-9.]+)')
        $hostPeaks = [regex]::Matches($text, 'host private=(\d+) MB peak=(\d+) MB')
        $workspaceLine=[regex]::Match($text,'(?m)^ntt_workspace_stats:.*').Value
        $workspace=@{}
        foreach($field in @('pool','hits','grows','mallocs','frees','workspace_bytes','big_bytes','small_bytes',
                           'table_bytes','owned_bytes','big_peak_bytes','small_peak_bytes','table_peak_bytes',
                           'owned_peak_bytes','aliases','alias_bytes','evictions','evicted_words','legacy_mallocs','legacy_frees',
                           'fuse_base_bytes','fuse_base_peak_bytes','full_bytes','full_peak_bytes')) {
            $m=[regex]::Match($workspaceLine,"(?:^| )$field=(\d+)")
            if($m.Success){$workspace[$field]=[UInt64]$m.Groups[1].Value}
        }
        $fuseLine=[regex]::Match($text,'(?m)^ntt_fuse_base_stats:.*').Value;$fuse=@{}
        $chunkLine=[regex]::Match($text,'(?m)^real_batched_chunkoutput:.*').Value;$chunk=@{}
        foreach($field in @('enabled','calls','reused_calls','whole_calls','multi_chunk_reused','reused_chunks','legacy_calls','grows','request_peak_bytes','whole_peak_bytes','retained_peak_bytes')) {
            $m=[regex]::Match($chunkLine,"(?:^| )$field=(\d+)")
            if($m.Success){$chunk[$field]=[UInt64]$m.Groups[1].Value}
        }
        if($chunk.Count -ne 11 -or "$($chunk.enabled)" -ne $env:NTT_S4_CHUNK_OUTPUT -or
           $chunk.calls -le 0 -or $chunk.calls -ne $chunk.reused_calls+$chunk.whole_calls -or
           $chunk.request_peak_bytes -gt $chunk.whole_peak_bytes -or $chunk.request_peak_bytes -gt $chunk.retained_peak_bytes -or
           ($chunk.enabled -eq 1 -and ($chunk.reused_calls -ne $chunk.calls -or $chunk.whole_calls -ne 0 -or
               ($Target -eq 'chunk_output' -and $D -gt 2310 -and $chunk.multi_chunk_reused -le 0))) -or
           ($chunk.enabled -eq 0 -and ($chunk.reused_calls -ne 0 -or $chunk.whole_calls -ne $chunk.calls))) {
            throw "Chunk output control/accounting failed; inspect $log"
        }
        $fullWall=[regex]::Match($text,'stage2_full_wall: curve=1 shape=([0-9.]+) init=([0-9.]+) main=([0-9.]+) total=([0-9.]+) fixture=([0-9.]+) clean=1')
        if(-not $fullWall.Success -or [math]::Abs([double]$fullWall.Groups[2].Value+[double]$fullWall.Groups[3].Value-[double]$fullWall.Groups[4].Value) -gt 0.000003) {
            throw "Full Stage2 timing boundary failed; inspect $log"
        }
        $windowLine=[regex]::Match($text,'(?m)^real_batched_outputwindow:.*').Value;$window=@{}
        foreach($field in @('enabled','calls','source_coeffs','reduced_coeffs','returned_coeffs','skipped_coeffs','d2h_words','device_peak_bytes','pinned_peak_bytes')) {
            $m=[regex]::Match($windowLine,"(?:^| )$field=(\d+)")
            if($m.Success){$window[$field]=[UInt64]$m.Groups[1].Value}
        }
        if($window.Count -ne 9 -or "$($window.enabled)" -ne $env:NTT_S4_OUTPUT_WINDOW -or
           $window.calls -le 0 -or $window.returned_coeffs -le 0 -or
           $window.source_coeffs -ne $window.reduced_coeffs+$window.skipped_coeffs -or
           $window.returned_coeffs -gt $window.reduced_coeffs -or
           ($window.enabled -eq 1 -and ($window.reduced_coeffs -ne $window.returned_coeffs -or $window.skipped_coeffs -le 0)) -or
           ($window.enabled -eq 0 -and $window.skipped_coeffs -ne 0)) {
            throw "Output window contract/accounting failed; inspect $log"
        }
        $finalLine=[regex]::Match($text,'(?m)^real_batched_finalreadback:.*').Value;$final=@{}
        foreach($field in @('enabled','calls','copied_words','avoided_words','host_peak_bytes','t_copy')) {
            $m=[regex]::Match($finalLine,"(?:^| )$field=([0-9.]+)")
            if($m.Success){$final[$field]=[double]$m.Groups[1].Value}
        }
        if($final.Count -ne 6 -or "$($final.enabled)" -ne $env:NTT_S4_FINAL_READBACK -or $final.calls -le 0 -or
           ($final.enabled -eq 1 -and ($final.copied_words -le 0 -or $final.avoided_words -ne 0 -or $final.host_peak_bytes -le 0)) -or
           ($final.enabled -eq 0 -and ($final.avoided_words -le 0 -or $final.copied_words -ne 0 -or $final.host_peak_bytes -ne 0 -or $final.t_copy -ne 0))) {
            throw "Final readback control/accounting failed; inspect $log"
        }
        foreach($field in @('compact','allocations','frees','live_bytes','peak_bytes')) {
            $m=[regex]::Match($fuseLine,"(?:^| )$field=(\d+)")
            if($m.Success){$fuse[$field]=[UInt64]$m.Groups[1].Value}
        }
        if($fuse.Count -ne 5 -or "$($fuse.compact)" -ne $env:NTT_FUSE_COMPACT_SCRATCH -or
           $fuse.live_bytes -ne $workspace.fuse_base_bytes -or $fuse.live_bytes -gt $fuse.peak_bytes) {
            throw "FuseCtx ownership/control accounting failed; inspect $log"
        }
        $naming = [regex]::Match($text, 'batched_naming:.*?t_scan=([0-9.]+) t_ladder=([0-9.]+) t_name=([0-9.]+)')
        $arenaSize = [regex]::Match($text, 'real_batched_breakdown:.*?ntt_seconds=([0-9.]+).*?arena_mb=([0-9.]+)')
        # This is the most recent giant-ladder snapshot, not an exact descent/naming minimum.
        $memorySnapshot = [regex]::Match($text, 'descent_begin:.*?device free=(\d+) MB of (\d+) MB')
        $oracleLine = [regex]::Match($text, '(?m)^s4_oracle_stats:.*').Value
        $oracle = @{}
        foreach ($field in @('async','selected','queued','compared','samples','pending','ring_waits',
                              'fallbacks','signature','t_wait','t_copy_host','t_gmp','host_total',
                              'pack','t_num','t_mod')) {
            $metric = [regex]::Match($oracleLine, ('(?:^| )' + $field + '=([^\s]+)'))
            if ($metric.Success) { $oracle[$field] = $metric.Groups[1].Value }
        }
        if ($rc -ne 0 -or -not $stage.Success -or -not $reduce.Success -or -not $arena.Success -or
            -not $phaseSplit.Success -or -not $naming.Success -or -not $arenaSize.Success -or
            $stage.Groups[2].Value -ne '0' -or $arena.Groups[1].Value -ne '0' -or
            $text -notmatch ("s4_reduce_mode: algorithm=" + $algorithm) -or
            $text -cmatch 'FATAL|MISMATCH|gmp_bad=[1-9]|gmp_selftest_bad=[1-9]|gmp_check_bad=[1-9]|mismatches=[1-9]|slot_canonical_bad=[1-9]') {
            throw "A/B validation failed (exit=$rc); inspect $log"
        }
        # Use the full stats line for the oracle sample count.
        $checked = [regex]::Match($text, 's4_multiply_stats:.*?gmp_checked=(\d+)')
        if ($oracle.Count -ne 16 -or $oracle.async -ne $oracleAsync -or $oracle.pack -ne $oraclePack -or
            $oracle.selected -ne $oracle.compared -or $oracle.pending -ne '0' -or
            $oracle.fallbacks -ne '0' -or -not $checked.Success -or
            $oracle.samples -ne $checked.Groups[1].Value -or
            ($oracleAsync -eq '1' -and $oracle.queued -ne $oracle.selected) -or
            ($oracleAsync -eq '0' -and $oracle.queued -ne '0')) {
            throw "oracle validation failed; inspect $log"
        }
        if (-not $carry.Success -or -not $upload.Success -or -not $transfers.Success -or
            -not $rawVolume.Success -or -not $backVolume.Success -or -not $carryTime.Success -or
            -not $carryLedger.Success -or
            [math]::Abs([double]$carryLedger.Groups[3].Value-[double]$carryLedger.Groups[1].Value-
                       [double]$carryLedger.Groups[2].Value) -gt 0.000002 -or
            $text -notmatch 'real_batched_carrytrace: enabled=0 words=0 ' -or
            $carry.Groups[4].Value -ne $carryBatch -or $carry.Groups[1].Value -ne $carry.Groups[5].Value -or
            [long]$carry.Groups[2].Value -gt [long]$carry.Groups[1].Value -or
            ([long]$carry.Groups[1].Value -gt 0 -and
             ([long]$carry.Groups[2].Value -lt 1 -or [long]$carry.Groups[6].Value -lt 1 -or
              ($carryBatch -eq '0' -and $carry.Groups[1].Value -ne $carry.Groups[2].Value))) -or
            ([long]$carry.Groups[6].Value -gt 1 -and $carry.Groups[1].Value -eq $carry.Groups[2].Value)) {
            throw "carry/staging validation failed; inspect $log"
        }
        if (-not $inputPack.Success -or $inputPack.Groups[1].Value -ne $packDirect -or
            [UInt64]$inputPack.Groups[6].Value -eq 0 -or
            ($packDirect -eq '1' -and ($inputPack.Groups[3].Value -ne '0' -or
                $inputPack.Groups[4].Value -ne '0' -or $inputPack.Groups[7].Value -ne '0' -or
                $inputPack.Groups[8].Value -ne '0' -or [long]$inputPack.Groups[2].Value -le 0 -or
                [UInt64]$inputPack.Groups[5].Value -eq 0 -or [double]$inputPack.Groups[10].Value -ne 0)) -or
            ($packDirect -eq '0' -and ($inputPack.Groups[2].Value -ne '0' -or
                $inputPack.Groups[5].Value -ne '0' -or [long]$inputPack.Groups[3].Value -le 0 -or
                [UInt64]$inputPack.Groups[4].Value -eq 0 -or
                $inputPack.Groups[6].Value -ne $inputPack.Groups[7].Value))) {
            throw "input pack validation failed; inspect $log"
        }
        if (-not $flat.Success -or $flat.Groups[1].Value -ne $flatDirect -or
            [long]$flat.Groups[3].Value+[long]$flat.Groups[4].Value -ne 2*[long]$flat.Groups[2].Value -or
            [long]$flat.Groups[5].Value -gt [long]$flat.Groups[4].Value -or
            [UInt64]$flat.Groups[10].Value -gt [UInt64]$flat.Groups[11].Value -or
            ($flatDirect -eq '0' -and ($flat.Groups[3].Value -ne '0' -or
                $flat.Groups[8].Value -ne '0' -or $flat.Groups[9].Value -ne '0' -or
                $flat.Groups[10].Value -ne $flat.Groups[11].Value))) {
            throw "flat input accounting failed; inspect $log"
        }
        if (-not $groot.Success -or $groot.Groups[1].Value -ne $grootOnly -or
            [long]$groot.Groups[2].Value -le 0 -or [UInt64]$groot.Groups[9].Value -eq 0 -or
            $groot.Groups[10].Value -ne '0' -or
            [UInt64]$groot.Groups[5].Value -lt [UInt64]$groot.Groups[6].Value -or
            ($grootOnly -eq '0' -and ($groot.Groups[3].Value -ne '0' -or $groot.Groups[4].Value -ne '0' -or
                $groot.Groups[7].Value -ne '0' -or $groot.Groups[8].Value -ne '0' -or
                $groot.Groups[5].Value -ne $groot.Groups[6].Value)) -or
            ($grootOnly -eq '1' -and -not $residentMode -and ([long]$groot.Groups[3].Value -le 0 -or
                [UInt64]$groot.Groups[7].Value -eq 0 -or [UInt64]$groot.Groups[8].Value -eq 0))) {
            throw "G-tree lifecycle accounting failed; inspect $log"
        }
        if($workspace.Count -ne 24 -or "$($workspace.pool)" -ne $env:NTT_ARENA_WORKSPACE_POOL -or
           $workspace.full_bytes -ne $workspace.owned_bytes+$workspace.fuse_base_bytes -or
           $workspace.full_bytes -gt $workspace.full_peak_bytes -or
           $workspace.fuse_base_bytes -gt $workspace.fuse_base_peak_bytes -or
           $workspace.big_bytes+$workspace.small_bytes+$workspace.table_bytes -ne $workspace.owned_bytes -or
           $workspace.workspace_bytes -gt $workspace.big_bytes -or $workspace.owned_bytes -gt $workspace.owned_peak_bytes -or
           $workspace.big_bytes -gt $workspace.big_peak_bytes -or $workspace.small_bytes -gt $workspace.small_peak_bytes -or
           $workspace.table_bytes -gt $workspace.table_peak_bytes -or $workspace.aliases -ne 0 -or $workspace.alias_bytes -ne 0 -or
           ($workspace.pool -eq 1 -and ($workspace.grows -lt 1 -or $workspace.hits -lt 1 -or
               $workspace.mallocs -ne 3*$workspace.grows -or $workspace.legacy_mallocs -ne 0)) -or
           ($workspace.pool -eq 0 -and ($workspace.workspace_bytes -ne 0 -or $workspace.hits -ne 0 -or
               $workspace.grows -ne 0 -or $workspace.mallocs -ne 0 -or $workspace.legacy_mallocs -lt 1))) {
            throw "workspace ownership/accounting validation failed; inspect $log"
        }
        $row = [pscustomobject]@{ run=$i+1; mode=$mode; elapsed=[double]$stage.Groups[5].Value;
            wall=[math]::Round($sw.Elapsed.TotalSeconds,3); t_reduce=[double]$reduce.Groups[2].Value;
            coeffs=[UInt64]$reduce.Groups[1].Value; hits=$stage.Groups[1].Value;
            factors=$stage.Groups[3].Value; hit_primes=$stage.Groups[4].Value; log=$log;
            started=$runStarted.ToString('o'); ended=$runEnded.ToString('o');
            oracle_samples=$oracle.samples; oracle_jobs=$oracle.selected; oracle_signature=$oracle.signature;
            oracle_wait=[double]$oracle.t_wait; oracle_copy_host=[double]$oracle.t_copy_host;
            oracle_gmp=[double]$oracle.t_gmp; oracle_host=[double]$oracle.host_total;
            oracle_num=[double]$oracle.t_num; oracle_mod=[double]$oracle.t_mod; oracle_pack=$oracle.pack;
            carry_deferred=$carry.Groups[1].Value; carry_finishes=$carry.Groups[2].Value;
            carry_slices=$carry.Groups[3].Value; carry_checked=$carry.Groups[5].Value;
            carry_max_group=$carry.Groups[6].Value; carry_batch=$carryBatch;
            raw_reuse_waits=$upload.Groups[1].Value; raw_reuse_wait=[double]$upload.Groups[2].Value;
            raw_pinned_bytes=$upload.Groups[3].Value; raw_async=$transfers.Groups[1].Value;
            out_async=$transfers.Groups[2].Value; raw_upload=[double]$rawVolume.Groups[1].Value;
            h2d_gib=[double]$rawVolume.Groups[2].Value; pack_launches=$rawVolume.Groups[3].Value;
            coeffback=[double]$backVolume.Groups[1].Value; d2h_gib=[double]$backVolume.Groups[2].Value;
            carry_d2h=[double]$carryTime.Groups[1].Value;
            carry_group_readback=[double]$carryLedger.Groups[1].Value;
            carry_chunk_readback=[double]$carryLedger.Groups[2].Value;
            carry_total_readback=[double]$carryLedger.Groups[3].Value;
            pack_direct=$packDirect; direct_chunks=$inputPack.Groups[2].Value;
            copied_chunks=$inputPack.Groups[3].Value; d2d_bytes=$inputPack.Groups[4].Value;
            avoided_bytes=$inputPack.Groups[5].Value; packed_peak_bytes=$inputPack.Groups[6].Value;
            temp_peak_bytes=$inputPack.Groups[7].Value; temp_current_bytes=$inputPack.Groups[8].Value;
            input_pack_host=[double]$inputPack.Groups[9].Value; input_copy_host=[double]$inputPack.Groups[10].Value;
            batch_mb=$batchBudget; giant=[double]$phaseSplit.Groups[1].Value;
            gtrees=[double]$phaseSplit.Groups[2].Value; fold=[double]$phaseSplit.Groups[3].Value;
            descent=[double]$phaseSplit.Groups[4].Value; inverse=[double]$phaseSplit.Groups[5].Value;
            accum=[double]$phaseSplit.Groups[6].Value; naming=[double]$phaseSplit.Groups[7].Value;
            f_tree=[double]$phaseSplit.Groups[8].Value; naming_scan=[double]$naming.Groups[1].Value;
            naming_ladder=[double]$naming.Groups[2].Value; ntt_seconds=[double]$arenaSize.Groups[1].Value;
            arena_mb=[double]$arenaSize.Groups[2].Value;
            giant_snapshot_free_mb=$(if ($memorySnapshot.Success) { [double]$memorySnapshot.Groups[1].Value } else { $null });
            flat_direct=$flatDirect; flat_calls=$flat.Groups[2].Value; flat_borrowed=$flat.Groups[3].Value;
            flat_padded=$flat.Groups[4].Value; flat_alias_clones=$flat.Groups[5].Value;
            flat_copy_bytes=$flat.Groups[6].Value; flat_zero_bytes=$flat.Groups[7].Value;
            flat_avoided_copy_bytes=$flat.Groups[8].Value; flat_avoided_zero_bytes=$flat.Groups[9].Value;
            flat_temp_peak_bytes=$flat.Groups[10].Value; flat_control_peak_bytes=$flat.Groups[11].Value;
            flat_prepare=[double]$flat.Groups[12].Value;
            groot_only=$grootOnly; groot_builds=$groot.Groups[2].Value; groot_nodes_released=$groot.Groups[3].Value;
            groot_moves=$groot.Groups[4].Value; groot_node_peak_bytes=$groot.Groups[5].Value;
            groot_retained_peak_bytes=$groot.Groups[6].Value; groot_released_bytes=$groot.Groups[7].Value;
            groot_input_released_bytes=$groot.Groups[8].Value; groot_root_words=$groot.Groups[9].Value;
            groot_release=[double]$groot.Groups[12].Value;
            observed_host_peak_mb=(@($hostPeaks | ForEach-Object {[double]$_.Groups[2].Value}) | Measure-Object -Maximum).Maximum }
        foreach($field in $small.Keys){$row|Add-Member -NotePropertyName ("small_"+$field) -NotePropertyValue $small[$field]}
        foreach($field in $seed.Keys){$row|Add-Member -NotePropertyName ("seed_"+$field) -NotePropertyValue $seed[$field]}
        foreach($field in $foldFlat.Keys){$row|Add-Member -NotePropertyName ("fold_flat_"+$field) -NotePropertyValue $foldFlat[$field]}
        foreach($field in $ginv.Keys){$row|Add-Member -NotePropertyName ("gfinv_"+$field) -NotePropertyValue $ginv[$field]}
        foreach($entry in @(@('gleaf_in',1),@('gleaf_inverse',2),@('gleaf_out',3),@('gscale',4))){
            $row|Add-Member -NotePropertyName $entry[0] -NotePropertyValue ([double]$gleaf.Groups[$entry[1]].Value)
        }
        foreach($entry in @(@('projective_leaves',1),@('gamma_points',2),@('projective_segments',3),@('affine_fallback_points',4))){
            $row|Add-Member -NotePropertyName $entry[0] -NotePropertyValue ([UInt64]$projective.Groups[$entry[1]].Value)
        }
        if($Target -eq 'gfinv_batch' -and (-not $gleaf.Success -or -not $projective.Success)){throw "Missing leaf preparation counters"}
        if($Target -eq 'gfinv_batch' -and $rows.Count -gt 0){
            foreach($field in @('gfinv_requests','projective_leaves','gamma_points','projective_segments','affine_fallback_points')) {
                if($row.$field -ne $rows[0].$field){throw "Segment/Gamma workload changed: $field"}
            }
        }
        foreach($field in $gmemory.Keys){$row|Add-Member -NotePropertyName ("gmemory_"+$field) -NotePropertyValue $gmemory[$field]}
        foreach($field in $gdevice.Keys){$row|Add-Member -NotePropertyName ("gdevice_"+$field) -NotePropertyValue $gdevice[$field]}
        $row|Add-Member -NotePropertyName groot_root_hash -NotePropertyValue $groot.Groups[11].Value
        $row|Add-Member -NotePropertyName leaf_count -NotePropertyValue $leaves.Groups[1].Value
        $row|Add-Member -NotePropertyName leaf_words -NotePropertyValue $leaves.Groups[2].Value
        $row|Add-Member -NotePropertyName leaf_hash -NotePropertyValue $leaves.Groups[3].Value
        foreach($pair in @(@('shape_p',1),@('shape_giant_points',2),@('shape_num_poly_g',3),@('shape_loops',4))) {
            $row|Add-Member -NotePropertyName $pair[0] -NotePropertyValue $shape.Groups[$pair[1]].Value
        }
        foreach($field in $scaled.Keys){$row|Add-Member -NotePropertyName ("scaled_"+$field) -NotePropertyValue $scaled[$field]}
        foreach($field in $workspace.Keys){$row|Add-Member -NotePropertyName ("workspace_"+$field) -NotePropertyValue $workspace[$field]}
        foreach($field in $fuse.Keys){$row|Add-Member -NotePropertyName ("fuse_"+$field) -NotePropertyValue $fuse[$field]}
        foreach($field in $final.Keys){$row|Add-Member -NotePropertyName ("final_"+$field) -NotePropertyValue $final[$field]}
        foreach($field in $window.Keys){$row|Add-Member -NotePropertyName ("window_"+$field) -NotePropertyValue $window[$field]}
        foreach($field in $chunk.Keys){$row|Add-Member -NotePropertyName ("chunk_"+$field) -NotePropertyValue $chunk[$field]}
        $row|Add-Member -NotePropertyName stage2_full -NotePropertyValue ([double]$fullWall.Groups[4].Value)
        $row|Add-Member -NotePropertyName stage2_init -NotePropertyValue ([double]$fullWall.Groups[2].Value)
        $row|Add-Member -NotePropertyName stage2_shape -NotePropertyValue ([double]$fullWall.Groups[1].Value)
        if($qHex) {
            $qSha=[Security.Cryptography.SHA256]::Create()
            try {$qHash=[BitConverter]::ToString($qSha.ComputeHash([Text.Encoding]::UTF8.GetBytes($qHex))).Replace('-','').ToLowerInvariant()} finally {$qSha.Dispose()}
        } else {$qHash=''}
        $row|Add-Member -NotePropertyName stage1_extra -NotePropertyValue $Stage1Extra
        $row|Add-Member -NotePropertyName q_sha256_hex -NotePropertyValue $qHash
        if($rows.Count -gt 0 -and $row.q_sha256_hex -ne $rows[0].q_sha256_hex){throw "Stage1 Q changed between modes; inspect $log"}
        if($residentMode -and $row.gdevice_trees -ne $row.groot_builds){throw "resident G tree/build count disagrees"}
        if($row.chunk_calls -ne $row.window_calls){throw "Chunk output logical calls disagree; inspect $log"}
        if($row.window_d2h_words+$row.gdevice_resident_words -ne $row.final_copied_words+$row.final_avoided_words -or
           $row.window_calls -ne $row.final_calls) {throw "Output/readback accounting disagrees; inspect $log"}
        if($rows.Count -gt 0) {
            foreach($field in @('groot_root_hash','leaf_count','leaf_words','leaf_hash','factors','hit_primes','hits',
                               'shape_p','shape_giant_points','shape_num_poly_g','shape_loops')) {
                if($row.$field -ne $rows[0].$field){throw "A/B output changed: $field; inspect $log"}
            }
        }
        # Scaled descent intentionally changes all internal multiply/transfer workloads.
        # Output equality is required above; per-run ledgers and within-mode repeats remain checked.
        if($Target -ne 'scaled_descent' -and $rows.Count -gt 0 -and ($row.final_calls -ne $rows[0].final_calls -or
           ($Target -ne 'output_window' -and
            $row.final_copied_words+$row.final_avoided_words -ne $rows[0].final_copied_words+$rows[0].final_avoided_words))) {
            throw "Final readback logical workload changed; inspect $log"
        }
        if($Target -ne 'scaled_descent' -and $rows.Count -gt 0 -and ($row.window_source_coeffs -ne $rows[0].window_source_coeffs -or
           $row.window_returned_coeffs -ne $rows[0].window_returned_coeffs)) {
            throw "Output window changed transform source or consumer workload; inspect $log"
        }
        if ($rows.Count -gt 0 -and ($row.groot_builds -ne $rows[0].groot_builds -or
            $row.groot_root_words -ne $rows[0].groot_root_words)) {
            throw "G-tree workload changed; inspect $log"
        }
        if ($Target -notin @('scaled_descent','fold_flat') -and $rows.Count -gt 0 -and ($row.flat_calls -ne $rows[0].flat_calls -or
            [UInt64]$row.flat_copy_bytes+[UInt64]$row.flat_avoided_copy_bytes -ne
            [UInt64]$rows[0].flat_copy_bytes+[UInt64]$rows[0].flat_avoided_copy_bytes -or
            [UInt64]$row.flat_zero_bytes+[UInt64]$row.flat_avoided_zero_bytes -ne
            [UInt64]$rows[0].flat_zero_bytes+[UInt64]$rows[0].flat_avoided_zero_bytes -or
            $row.flat_control_peak_bytes -ne $rows[0].flat_control_peak_bytes)) {
            throw "flat input logical workload changed; inspect $log"
        }
        if ($Target -ne 'scaled_descent' -and $rows.Count -gt 0 -and (($Target -ne 'output_window' -and $row.coeffs -ne $rows[0].coeffs) -or $row.factors -ne $rows[0].factors -or
            $row.hit_primes -ne $rows[0].hit_primes -or $row.hits -ne $rows[0].hits -or
            ($Target -ne 'batch_mb' -and (
            ($Target -ne 'output_window' -and ($row.oracle_samples -ne $rows[0].oracle_samples -or
            $row.oracle_jobs -ne $rows[0].oracle_jobs -or $row.oracle_signature -ne $rows[0].oracle_signature)) -or
            $row.carry_deferred -ne $rows[0].carry_deferred -or $row.carry_slices -ne $rows[0].carry_slices -or
            ($Target -ne 'groot_device' -and ($row.raw_async -ne $rows[0].raw_async -or $row.out_async -ne $rows[0].out_async)) -or
            $row.pack_launches -ne $rows[0].pack_launches)) -or
            ($Target -ne 'groot_device' -and ($row.h2d_gib -ne $rows[0].h2d_gib -or ($Target -ne 'output_window' -and $row.d2h_gib -ne $rows[0].d2h_gib))))) {
            throw "A/B results or coefficient count changed; inspect $log"
        }
        if ($Target -ne 'scaled_descent' -and $rows.Count -gt 0 -and
            ([UInt64]$row.d2d_bytes+[UInt64]$row.avoided_bytes -ne
             [UInt64]$rows[0].d2d_bytes+[UInt64]$rows[0].avoided_bytes -or
             ($Target -ne 'batch_mb' -and (
             [long]$row.direct_chunks+[long]$row.copied_chunks -ne
             [long]$rows[0].direct_chunks+[long]$rows[0].copied_chunks -or
             $row.packed_peak_bytes -ne $rows[0].packed_peak_bytes)))) {
            throw "A/B packed input workload changed; inspect $log"
        }
        # Budget changes intentionally change chunk/sample schedules. Repeated runs of EACH
        # mode must still reproduce the complete sampling and staging workload.
        $sameMode = @($rows | Where-Object mode -eq $mode)
        if ($sameMode.Count) {
            foreach ($field in @('oracle_samples','oracle_jobs','oracle_signature','carry_deferred',
                                 'carry_slices','carry_checked','carry_finishes','carry_max_group',
                                 'raw_async','out_async','raw_pinned_bytes','pack_launches','direct_chunks','copied_chunks',
                                 'packed_peak_bytes','temp_peak_bytes','batch_mb','flat_calls','flat_borrowed',
                                 'flat_padded','flat_alias_clones','flat_copy_bytes','flat_zero_bytes',
                                 'flat_avoided_copy_bytes','flat_avoided_zero_bytes','flat_temp_peak_bytes',
                                 'groot_nodes_released','groot_moves','groot_node_peak_bytes','groot_retained_peak_bytes',
                                 'groot_released_bytes','groot_input_released_bytes')) {
                if ($row.$field -ne $sameMode[0].$field) { throw "repeated $mode changed $field; inspect $log" }
            }
        }
        if($sameMode.Count){foreach($field in $ginv.Keys){if($field -ne 't_prepare'){
            $name="gfinv_"+$field;if($row.$name -ne $sameMode[0].$name){throw "Segment inverse workload changed within mode: $field"}
        }}}
        if($sameMode.Count){foreach($field in $gmemory.Keys){$name="gmemory_"+$field;if($row.$name -ne $sameMode[0].$name){throw "G-root memory workload changed within mode: $field"}}}
        if($sameMode.Count){foreach($field in $gdevice.Keys){$name="gdevice_"+$field;if($row.$name -ne $sameMode[0].$name){throw "resident G workload changed within mode: $field"}}}
        if($sameMode.Count){foreach($field in $scaled.Keys){$name="scaled_"+$field;if($row.$name -ne $sameMode[0].$name){throw "scaled workload changed within mode: $field"}}}
        if($sameMode.Count){foreach($field in $workspace.Keys){$name="workspace_"+$field;if($row.$name -ne $sameMode[0].$name){throw "workspace workload changed within mode: $field"}}}
        if($sameMode.Count){foreach($field in $fuse.Keys){$name="fuse_"+$field;if($row.$name -ne $sameMode[0].$name){throw "fuse workload changed within mode: $field"}}}
        if($sameMode.Count){foreach($field in @('calls','copied_words','avoided_words','host_peak_bytes')){$name="final_"+$field;if($row.$name -ne $sameMode[0].$name){throw "final readback workload changed within mode: $field"}}}
        if($sameMode.Count){foreach($field in $window.Keys){$name="window_"+$field;if($row.$name -ne $sameMode[0].$name){throw "output window workload changed within mode: $field"}}}
        if($sameMode.Count){foreach($field in $chunk.Keys){$name="chunk_"+$field;if($row.$name -ne $sameMode[0].$name){throw "chunk output workload changed within mode: $field"}}}
        if($sameMode.Count -and ($row.coeffs -ne $sameMode[0].coeffs -or $row.d2h_gib -ne $sameMode[0].d2h_gib)) {
            throw "repeated $mode changed reduction/transfer workload; inspect $log"
        }
        $rows += $row
        $rows | Export-Csv -LiteralPath (Join-Path $Output 'results.csv') -NoTypeInformation -Encoding UTF8
        Write-Host ("  elapsed={0:F2}s wall={1:F2}s t_reduce={2:F3}s coeffs={3}" -f
                    $row.elapsed, $row.wall, $row.t_reduce, $row.coeffs)
    }
    $old = $rows | Where-Object mode -eq $order[0]
    $new = $rows | Where-Object mode -eq $order[1]
    $oldTime = ($old | Measure-Object elapsed -Average).Average
    $newTime = ($new | Measure-Object elapsed -Average).Average
    $oldReduce = ($old | Measure-Object t_reduce -Average).Average
    $newReduce = ($new | Measure-Object t_reduce -Average).Average
    Write-Host ("ABBA mean: elapsed {0:F2} -> {1:F2}s ({2:F1}%); t_reduce {3:F3} -> {4:F3}s ({5:F1}%)" -f
                $oldTime, $newTime, (100*($newTime/$oldTime-1)), $oldReduce, $newReduce, (100*($newReduce/$oldReduce-1)))
    $oldFull = ($old | Measure-Object stage2_full -Average).Average
    $newFull = ($new | Measure-Object stage2_full -Average).Average
    Write-Host ("full Stage2 (init + main, excludes Stage1): {0:F6} -> {1:F6}s ({2:F2}%)" -f
                $oldFull, $newFull, (100*($newFull/$oldFull-1)))
    Write-Host ("phase means: Gtree {0:F3} -> {1:F3}s; fold {2:F3} -> {3:F3}s; descent {4:F3} -> {5:F3}s" -f
                ($old | Measure-Object gtrees -Average).Average, ($new | Measure-Object gtrees -Average).Average,
                ($old | Measure-Object fold -Average).Average, ($new | Measure-Object fold -Average).Average,
                ($old | Measure-Object descent -Average).Average, ($new | Measure-Object descent -Average).Average)
    Write-Host ("oracle host: {0:F3} -> {1:F3}s; wait {2:F3} -> {3:F3}s; GMP {4:F3} -> {5:F3}s" -f
                ($old | Measure-Object oracle_host -Average).Average,
                ($new | Measure-Object oracle_host -Average).Average,
                ($old | Measure-Object oracle_wait -Average).Average,
                ($new | Measure-Object oracle_wait -Average).Average,
                ($old | Measure-Object oracle_gmp -Average).Average,
                ($new | Measure-Object oracle_gmp -Average).Average)
    Write-Host ("carry finishes: {0:F0} -> {1:F0}; checked interior chunks: {2:F0} -> {3:F0}" -f
                ($old | Measure-Object carry_finishes -Average).Average,
                ($new | Measure-Object carry_finishes -Average).Average,
                ($old | Measure-Object carry_checked -Average).Average,
                ($new | Measure-Object carry_checked -Average).Average)
    Write-Host ("packed input D2D: {0:F3} -> {1:F3} GiB; temporary device inputs: {2:F1} -> {3:F1} MiB" -f
                (($old | Measure-Object d2d_bytes -Average).Average / 1GB),
                (($new | Measure-Object d2d_bytes -Average).Average / 1GB),
                (($old | Measure-Object temp_peak_bytes -Average).Average / 1MB),
                (($new | Measure-Object temp_peak_bytes -Average).Average / 1MB))
    Write-Host ("pack launches: {0:F0} -> {1:F0}; arena: {2:F1} -> {3:F1} MiB; naming ladder: {4:F3} -> {5:F3}s" -f
                ($old | Measure-Object pack_launches -Average).Average,
                ($new | Measure-Object pack_launches -Average).Average,
                ($old | Measure-Object arena_mb -Average).Average,
                ($new | Measure-Object arena_mb -Average).Average,
                ($old | Measure-Object naming_ladder -Average).Average,
                ($new | Measure-Object naming_ladder -Average).Average)
    Write-Host ("flat host preparation: {0:F3} -> {1:F3}s; copied {2:F3} -> {3:F3} GiB; zeroed {4:F3} -> {5:F3} GiB; padding peak {6:F1} -> {7:F1} MiB" -f
                ($old | Measure-Object flat_prepare -Average).Average,
                ($new | Measure-Object flat_prepare -Average).Average,
                (($old | Measure-Object flat_copy_bytes -Average).Average / 1GB),
                (($new | Measure-Object flat_copy_bytes -Average).Average / 1GB),
                (($old | Measure-Object flat_zero_bytes -Average).Average / 1GB),
                (($new | Measure-Object flat_zero_bytes -Average).Average / 1GB),
                (($old | Measure-Object flat_temp_peak_bytes -Average).Average / 1MB),
                (($new | Measure-Object flat_temp_peak_bytes -Average).Average / 1MB))
    Write-Host ("G-tree node capacities: peak {0:F1} -> {1:F1} MiB; retained {2:F1} -> {3:F1} MiB; observed process host peak {4:F0} -> {5:F0} MB" -f
                (($old | Measure-Object groot_node_peak_bytes -Average).Average / 1MB),
                (($new | Measure-Object groot_node_peak_bytes -Average).Average / 1MB),
                (($old | Measure-Object groot_retained_peak_bytes -Average).Average / 1MB),
                (($new | Measure-Object groot_retained_peak_bytes -Average).Average / 1MB),
                ($old | Measure-Object observed_host_peak_mb -Average).Average,
                ($new | Measure-Object observed_host_peak_mb -Average).Average)
    Write-Host ("NTT cached payload peak: {0:F1} -> {1:F1} MiB; A/B/Q peak: {2:F1} -> {3:F1} MiB; table peak: {4:F1} -> {5:F1} MiB" -f
                (($old | Measure-Object workspace_owned_peak_bytes -Average).Average / 1MB),
                (($new | Measure-Object workspace_owned_peak_bytes -Average).Average / 1MB),
                (($old | Measure-Object workspace_big_peak_bytes -Average).Average / 1MB),
                (($new | Measure-Object workspace_big_peak_bytes -Average).Average / 1MB),
                (($old | Measure-Object workspace_table_peak_bytes -Average).Average / 1MB),
                (($new | Measure-Object workspace_table_peak_bytes -Average).Average / 1MB))
    Write-Host ("workspace CUDA allocations: {0:F0} -> {1:F0}; capacity growths: {2:F0}; reuse hits: {3:F0}" -f
                (($old | Measure-Object workspace_legacy_mallocs -Average).Average + ($old | Measure-Object workspace_mallocs -Average).Average),
                (($new | Measure-Object workspace_legacy_mallocs -Average).Average + ($new | Measure-Object workspace_mallocs -Average).Average),
                ($new | Measure-Object workspace_grows -Average).Average,
                ($new | Measure-Object workspace_hits -Average).Average)
    Write-Host ("FuseCtx mandatory payload peak: {0:F1} -> {1:F1} MiB; complete cached arena payload peak: {2:F1} -> {3:F1} MiB" -f
                (($old | Measure-Object workspace_fuse_base_peak_bytes -Average).Average / 1MB),
                (($new | Measure-Object workspace_fuse_base_peak_bytes -Average).Average / 1MB),
                (($old | Measure-Object workspace_full_peak_bytes -Average).Average / 1MB),
                (($new | Measure-Object workspace_full_peak_bytes -Average).Average / 1MB))
    Write-Host ("additional whole-call D2H (ALL S4 calls incl F-tree, excluded from chunk coeffback): {0:F3} -> {1:F3} GiB; copy host time: {2:F3} -> {3:F3}s; temporary host payload peak: {4:F1} -> {5:F1} MiB" -f
                (($old | Measure-Object final_copied_words -Average).Average * 8 / 1GB),
                (($new | Measure-Object final_copied_words -Average).Average * 8 / 1GB),
                ($old | Measure-Object final_t_copy -Average).Average,
                ($new | Measure-Object final_t_copy -Average).Average,
                (($old | Measure-Object final_host_peak_bytes -Average).Average / 1MB),
                (($new | Measure-Object final_host_peak_bytes -Average).Average / 1MB))
    Write-Host ("output window (ALL S4 incl F-tree): reduced {0:F0} -> {1:F0} coeffs; chunk D2H {2:F3} -> {3:F3} GiB; retained output device capacity {4:F1} -> {5:F1} MiB; pinned capacity {6:F1} -> {7:F1} MiB" -f
                ($old | Measure-Object window_reduced_coeffs -Average).Average,
                ($new | Measure-Object window_reduced_coeffs -Average).Average,
                (($old | Measure-Object window_d2h_words -Average).Average * 8 / 1GB),
                (($new | Measure-Object window_d2h_words -Average).Average * 8 / 1GB),
                (($old | Measure-Object window_device_peak_bytes -Average).Average / 1MB),
                (($new | Measure-Object window_device_peak_bytes -Average).Average / 1MB),
                (($old | Measure-Object window_pinned_peak_bytes -Average).Average / 1MB),
                (($new | Measure-Object window_pinned_peak_bytes -Average).Average / 1MB))
} finally {
    foreach ($key in $saved.Keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') }
}
