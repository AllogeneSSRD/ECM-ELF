#Requires -Version 5.1
<#
.SYNOPSIS
    Same-binary ABBA benchmark of reduction, sampling oracle, carry checks, packing or chunk budget.
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
    [int]$Device = 1,
    [ValidateSet('reduction','oracle','oracle_pack','carry_batch','pack_direct','batch_mb','flat_direct','groot','workspace')][string]$Target = 'reduction',
    [ValidateRange(1,256)][int]$BatchMB = 32,
    [ValidateRange(1,256)][int]$CandidateBatchMB = 64,
    [ValidateRange(0,65536)][int]$ArenaMB = 0,
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
if ($Target -eq 'carry_batch') { $order = @('carry_per_chunk', 'carry_batch', 'carry_batch', 'carry_per_chunk') }
if ($Target -eq 'pack_direct') { $order = @('pack_copy', 'pack_direct', 'pack_direct', 'pack_copy') }
if ($Target -eq 'batch_mb') { $order = @("batch_$BatchMB", "batch_$CandidateBatchMB", "batch_$CandidateBatchMB", "batch_$BatchMB") }
if ($Target -eq 'flat_direct') { $order = @('flat_copy','flat_direct','flat_direct','flat_copy') }
if ($Target -eq 'groot') { $order = @('full_gtree','groot','groot','full_gtree') }
if ($Target -eq 'workspace') { $order = @('keyed_workspace','workspace_pool','workspace_pool','keyed_workspace') }
$overrides = @{ NTT_NAME_MAX='1'; NTT_S4_BATCH_MB="$BatchMB"; NTT_S4_ASYNC='1';
                NTT_S4_DEFER_CARRY='1'; NTT_S4_HOSTPACK='0'; NTT_S5_ON='0'; NTT_S4_OLDTAIL='1';
                NTT_S5_REDDUMP='0'; NTT_S4_ORACLE_ASYNC='0'; NTT_S4_ORACLE_RING='4'; NTT_S4_ORACLE_PACK='1';
                NTT_S4_ORACLE_TEST_BAD='0'; NTT_S4_SAMPLE='96'; NTT_S4_CHECK_EVERY='8';
                NTT_S4_CARRY_BATCH='0'; NTT_S4_CHUNK_MAX='0'; NTT_S4_CARRY_TEST_BAD='0';
                NTT_S4_CARRY_TRACE='0'; NTT_S4_PACK_DIRECT='1'; NTT_S4_FLAT_DIRECT='1'; NTT_S4_FLAT_TEST='0';
                NTT_CARRY_ROUNDS=''; NTT_S4_GROOT_ONLY='1'; NTT_S4_GROOT_TEST='0'; NTT_S4_OFF='0';
                NTT_ARENA_WORKSPACE_POOL='1'; NTT_ARENA_WORKSPACE_TEST='0' }
if ($ArenaMB -gt 0) { $overrides.NTT_ARENA_CAP_KB = "$([long]$ArenaMB * 1024)" }
$saved = @{}
foreach ($key in $overrides.Keys) { $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
$modeControls = @(foreach ($mode in $order) {
    [pscustomobject]@{ mode=$mode;
        NTT_S4_OLDTAIL=$(if ($mode -eq 'montgomery') { '1' } else { '0' });
        NTT_S4_ORACLE_ASYNC=$(if ($mode -eq 'oracle_async') { '1' } else { '0' });
        NTT_S4_ORACLE_PACK=$(if ($mode -eq 'gmp_digits') { '0' } else { '1' });
        NTT_S4_CARRY_BATCH=$(if ($mode -eq 'carry_batch') { '1' } else { '0' });
        NTT_S4_PACK_DIRECT=$(if ($Target -eq 'pack_direct' -and $mode -eq 'pack_copy') { '0' } else { '1' });
        NTT_S4_FLAT_DIRECT=$(if ($mode -eq 'flat_copy') { '0' } else { '1' });
        NTT_S4_GROOT_ONLY=$(if ($mode -eq 'full_gtree') { '0' } else { '1' });
        NTT_ARENA_WORKSPACE_POOL=$(if ($mode -eq 'keyed_workspace') { '0' } else { '1' });
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
        $algorithm = $(if ($mode -eq 'montgomery') { 'montgomery' } else { 'division' })
        $oracleAsync = $(if ($mode -eq 'oracle_async') { '1' } else { '0' })
        $oraclePack = $(if ($mode -eq 'gmp_digits') { '0' } else { '1' })
        $carryBatch = $(if ($mode -eq 'carry_batch') { '1' } else { '0' })
        $packDirect = $(if ($Target -eq 'pack_direct' -and $mode -eq 'pack_copy') { '0' } else { '1' })
        $batchBudget = $(if ($Target -eq 'batch_mb' -and $mode -eq "batch_$CandidateBatchMB") { $CandidateBatchMB } else { $BatchMB })
        $flatDirect = $(if ($mode -eq 'flat_copy') { '0' } else { '1' })
        $grootOnly = $(if ($mode -eq 'full_gtree') { '0' } else { '1' })
        $env:NTT_S4_OLDTAIL = $(if ($algorithm -eq 'montgomery') { '1' } else { '0' })
        $env:NTT_S4_ORACLE_ASYNC = $oracleAsync
        $env:NTT_S4_ORACLE_PACK = $oraclePack
        $env:NTT_S4_CARRY_BATCH = $carryBatch
        $env:NTT_S4_PACK_DIRECT = $packDirect
        $env:NTT_S4_BATCH_MB = "$batchBudget"
        $env:NTT_S4_FLAT_DIRECT = $flatDirect
        $env:NTT_S4_GROOT_ONLY = $grootOnly
        $env:NTT_ARENA_WORKSPACE_POOL = $(if ($mode -eq 'keyed_workspace') { '0' } else { '1' })
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
        $stage = [regex]::Match($text, 'stage2:.*?hits=(\d+) bad_factors=(\d+) factors=([^\s]*) hit_primes=([^\s]*) elapsed=([0-9.]+)')
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
                           'owned_peak_bytes','aliases','alias_bytes','evictions','evicted_words','legacy_mallocs','legacy_frees')) {
            $m=[regex]::Match($workspaceLine,"(?:^| )$field=(\d+)")
            if($m.Success){$workspace[$field]=[UInt64]$m.Groups[1].Value}
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
            ($grootOnly -eq '1' -and ([long]$groot.Groups[3].Value -le 0 -or
                [UInt64]$groot.Groups[7].Value -eq 0 -or [UInt64]$groot.Groups[8].Value -eq 0))) {
            throw "G-tree lifecycle accounting failed; inspect $log"
        }
        if($workspace.Count -ne 20 -or "$($workspace.pool)" -ne $env:NTT_ARENA_WORKSPACE_POOL -or
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
        foreach($field in $workspace.Keys){$row|Add-Member -NotePropertyName ("workspace_"+$field) -NotePropertyValue $workspace[$field]}
        if ($rows.Count -gt 0 -and ($row.groot_builds -ne $rows[0].groot_builds -or
            $row.groot_root_words -ne $rows[0].groot_root_words)) {
            throw "G-tree workload changed; inspect $log"
        }
        if ($rows.Count -gt 0 -and ($row.flat_calls -ne $rows[0].flat_calls -or
            [UInt64]$row.flat_copy_bytes+[UInt64]$row.flat_avoided_copy_bytes -ne
            [UInt64]$rows[0].flat_copy_bytes+[UInt64]$rows[0].flat_avoided_copy_bytes -or
            [UInt64]$row.flat_zero_bytes+[UInt64]$row.flat_avoided_zero_bytes -ne
            [UInt64]$rows[0].flat_zero_bytes+[UInt64]$rows[0].flat_avoided_zero_bytes -or
            $row.flat_control_peak_bytes -ne $rows[0].flat_control_peak_bytes)) {
            throw "flat input logical workload changed; inspect $log"
        }
        if ($rows.Count -gt 0 -and ($row.coeffs -ne $rows[0].coeffs -or $row.factors -ne $rows[0].factors -or
            $row.hit_primes -ne $rows[0].hit_primes -or $row.hits -ne $rows[0].hits -or
            ($Target -ne 'batch_mb' -and (
            $row.oracle_samples -ne $rows[0].oracle_samples -or $row.oracle_jobs -ne $rows[0].oracle_jobs -or
            $row.oracle_signature -ne $rows[0].oracle_signature -or
            $row.carry_deferred -ne $rows[0].carry_deferred -or $row.carry_slices -ne $rows[0].carry_slices -or
            $row.raw_async -ne $rows[0].raw_async -or $row.out_async -ne $rows[0].out_async -or
            $row.pack_launches -ne $rows[0].pack_launches)) -or
            $row.h2d_gib -ne $rows[0].h2d_gib -or $row.d2h_gib -ne $rows[0].d2h_gib)) {
            throw "A/B results or coefficient count changed; inspect $log"
        }
        if ($rows.Count -gt 0 -and
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
                                 'raw_async','out_async','pack_launches','direct_chunks','copied_chunks',
                                 'packed_peak_bytes','temp_peak_bytes','batch_mb','flat_calls','flat_borrowed',
                                 'flat_padded','flat_alias_clones','flat_copy_bytes','flat_zero_bytes',
                                 'flat_avoided_copy_bytes','flat_avoided_zero_bytes','flat_temp_peak_bytes',
                                 'groot_nodes_released','groot_moves','groot_node_peak_bytes','groot_retained_peak_bytes',
                                 'groot_released_bytes','groot_input_released_bytes')) {
                if ($row.$field -ne $sameMode[0].$field) { throw "repeated $mode changed $field; inspect $log" }
            }
        }
        if($sameMode.Count){foreach($field in $workspace.Keys){$name="workspace_"+$field;if($row.$name -ne $sameMode[0].$name){throw "workspace workload changed within mode: $field"}}}
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
                ($old | Measure-Object workspace_legacy_mallocs -Average).Average,
                ($new | Measure-Object workspace_mallocs -Average).Average,
                ($new | Measure-Object workspace_grows -Average).Average,
                ($new | Measure-Object workspace_hits -Average).Average)
} finally {
    foreach ($key in $saved.Keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') }
}
